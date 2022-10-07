import torch
import torch.nn as nn
from torch.nn import SyncBatchNorm
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from math import sqrt
from collections import OrderedDict
from data.datasets import get_dataloader
from training.utils import get_optimizer_scheduler
from models import SupervisedModel, SelfSupervisedModel, Predictor, get_backbone
from loss.embedding_loss import EmbeddingLoss
from loss.simsiam_loss import SimSiamLoss


class Trainer(nn.Module):
    def __init__(self, config, views, writer, log_frequency, distributed, device, world_size=1, rank=0):
        super(Trainer, self).__init__()

        self.config = config
        self.distributed = distributed
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.dataloader = get_dataloader(self.config,
                                         mode='train', split='train',
                                         device=self.device,
                                         return_view_loader=views,
                                         rank=self.rank,
                                         world_size=self.world_size)
        self.batch_size = self.dataloader.batch_size
        self.iterator = iter(self.dataloader)

        self.scaler = GradScaler()

        self.step_count = 0
        self.writer = writer
        self.log_frequency = log_frequency

    def step(self, iteration):
        raise NotImplementedError


class SupervisedTrainer(Trainer):
    def __init__(self, config, backbone, writer, distributed, device, world_size=1, rank=0):
        super(SupervisedTrainer, self).__init__(config=config,
                                                views=False,
                                                writer=writer,
                                                log_frequency=config['LOG']['SUPERVISED_LOG'],
                                                distributed=distributed,
                                                device=device,
                                                rank=rank,
                                                world_size=world_size)

        self.model = SupervisedModel(backbone=backbone,
                                     embedding_head=config['MODEL']['EMBEDDING_HEAD'],
                                     embedding_dimension=config['MODEL']['EMBEDDING_DIMENSION'],
                                     boundary_head=config['MODEL']['BOUNDARY_HEAD'],
                                     norm=config['MODEL']['NORM'],
                                     padding_mode=config['MODEL']['PADDING']).to(device)
        if config['supervised_checkpoint']:
            self.model.load_state_dict(torch.load(config['supervised_checkpoint'], map_location=device))

        elif config['ssl_checkpoint']:
            state_dict = torch.load(config['ssl_checkpoint'], map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.backbone' in k:
                    name = k.replace('module.backbone.', '')  # remove 'module.' of DataParallel/DistributedDataParallel and only use the backbone
                    new_state_dict[name] = v
            self.model.backbone.load_state_dict(new_state_dict)

        if distributed:
            if config['MODEL']['NORM'] == 'batch_norm':
                self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        self.optimizer, self.scheduler = get_optimizer_scheduler(config, self.model.parameters())

        self.criterion = EmbeddingLoss(alpha=config['LOSS']['ALPHA'],
                                       beta=config['LOSS']['BETA'],
                                       gamma=config['LOSS']['GAMMA'],
                                       delta=config['LOSS']['DELTA'],
                                       device=device,
                                       average=1)

    def step(self, iteration):
        # initialize model for producing targets
        self.model.train()

        # data loading
        pos, volume, target, _ = next(self.iterator)
        X = volume.to(self.device, non_blocking=True)
        y_instances = target[0].to(self.device)
        y_foreground_mask = target[1].to(self.device)

        # forward and backward pass
        self.optimizer.zero_grad()
        with autocast():
            output = self.model(X)
            loss = self.criterion(embeddings=output.embeddings,
                                  masks=output.boundaries,
                                  target_embeddings=y_instances,
                                  target_masks=y_foreground_mask)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.rank == 0 and iteration > 0 and self.step_count % self.log_frequency == 0:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f"Supervised {self.step_count} "
                  f"lr: {lr:.2e} "
                  f"Total Loss: {loss.item():.3e} ")

            # write to tensorboard log
            self.writer.step(iteration,
                             images=X,
                             target_masks=y_foreground_mask,
                             embedding_pred=output.embeddings,
                             mask_pred=output.boundaries,
                             criterion=self.criterion,
                             model_params=self.model.parameters(),
                             lr=lr)

        self.step_count += 1


class SimSiamTrainer(Trainer):
    def __init__(self, config, backbone, encoder_only, writer, distributed, device, world_size=1, rank=0):
        super(SimSiamTrainer, self).__init__(config=config,
                                             views=True,
                                             writer=writer,
                                             log_frequency=config['LOG']['UNSUPERVISED_LOG'],
                                             distributed=distributed,
                                             device=device,
                                             rank=rank,
                                             world_size=world_size)

        self.model = SelfSupervisedModel(backbone=backbone,
                                         feature_pooling_kernel_size=config['SSL']['PROJECTION_POOLING'],
                                         projection_dim=config['SSL']['PROJECTION_DIMENSION'],
                                         projection_head=config['SSL']['PROJECTION_HEAD'],
                                         encoder_only=encoder_only).to(device)

        self.predictor = Predictor(dim=config['SSL']['PROJECTION_DIMENSION'],
                                   pred_dim=config['SSL']['PROJECTION_DIMENSION']).to(device)

        if config['ssl_checkpoint']:
            self.model.load_state_dict(torch.load(config['ssl_checkpoint'], map_location=device))
            self.predictor.load_state_dict(torch.load(config['predictor_checkpoint'], map_location=device))

        if distributed:
            if config['MODEL']['NORM'] == 'batch_norm':
                self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.predictor = SyncBatchNorm.convert_sync_batchnorm(self.predictor)
            self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            self.predictor = DDP(self.predictor, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        parameters = [{'params': self.model.parameters()},
                      {'params': self.predictor.parameters(), 'lr': config['OPTIM']['PREDICTOR_LR']}]

        self.optimizer, self.scheduler = get_optimizer_scheduler(config, parameters)

        self.criterion = SimSiamLoss(dim=1, eps=1e-8)

    def step(self, iteration):
        self.model.train()
        self.predictor.train()

        # data loading
        pos, views_a, views_b = next(self.iterator)
        X1, X2 = views_a.to(self.device, non_blocking=True), views_b.to(self.device, non_blocking=True)

        # forward and backward pass
        self.optimizer.zero_grad()
        with autocast():
            y1, z1 = self.model(X1)
            y2, z2 = self.model(X2)

            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

            loss = self.criterion(z1=z1.detach(), z2=z2.detach(), p1=p1, p2=p2)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.rank == 0 and self.step_count % self.log_frequency == 0:
            zcat = torch.cat([z1, z2])
            std = torch.mean(torch.std(zcat / torch.norm(zcat, dim=1, keepdim=True), dim=0))
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(40 * '*')
            print(f"SSL {self.step_count} Loss {loss.item():.3f} "
                  f"std(z/|z|) {std.item():.4f} "
                  f"sqrt(1/d) {1 / sqrt(z1.shape[1]):.4f} "
                  f"y norm {torch.mean(torch.norm(y1, dim=1)).item():.4f} "
                  f"z norm {torch.mean(torch.norm(z1, dim=1)).item():.4f} ")
            self.writer.ssl_step(iteration, X1, X2, y1, y2, z1, z2, loss.item(), self.model.parameters(), lr)

        self.step_count += 1


class MoCoTrainer(Trainer):
    """
    CODE broadly copied from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """
    def __init__(self, config, backbone, encoder_only, writer, distributed, device, world_size=1, rank=0):
        super(MoCoTrainer, self).__init__(config=config,
                                          views=True,
                                          writer=writer,
                                          log_frequency=config['LOG']['UNSUPERVISED_LOG'],
                                          distributed=distributed,
                                          device=device,
                                          rank=rank,
                                          world_size=world_size)
        assert config['SSL']['MOMENTUM'] and config['SSL']['DICT_SIZE'] and config['SSL']['TEMPERATURE']
        self.m = config['SSL']['MOMENTUM']
        self.K = config['SSL']['DICT_SIZE']
        self.T = config['SSL']['TEMPERATURE']

        self.model = SelfSupervisedModel(backbone=backbone,
                                         feature_pooling_kernel_size=config['SSL']['PROJECTION_POOLING'],
                                         projection_dim=config['SSL']['PROJECTION_DIMENSION'],
                                         projection_head=config['SSL']['PROJECTION_HEAD'],
                                         encoder_only=encoder_only).to(device)

        key_backbone = get_backbone(config)
        self.encoder_k = SelfSupervisedModel(backbone=key_backbone,
                                             feature_pooling_kernel_size=config['SSL']['PROJECTION_POOLING'],
                                             projection_dim=config['SSL']['PROJECTION_DIMENSION'],
                                             projection_head=config['SSL']['PROJECTION_HEAD'],
                                             encoder_only=encoder_only).to(device)

        if config['ssl_checkpoint']:
            state_dict = torch.load(config['ssl_checkpoint'], map_location=device)
            if world_size > 1:
                self.model.load_state_dict(state_dict)
            else:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                    new_state_dict[name] = v

                self.model.load_state_dict(new_state_dict)

        if distributed:
            if config['MODEL']['NORM'] == 'batch_norm':
                self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.encoder_k = SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
            self.model = DDP(self.model, device_ids=[rank], output_device=rank,
                                 find_unused_parameters=False)
            self.encoder_k = DDP(self.encoder_k, device_ids=[rank], output_device=rank,
                                 find_unused_parameters=False)

        for param_q, param_k in zip(self.model.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(config['SSL']['PROJECTION_DIMENSION'], self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0).to(device)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.optimizer, self.scheduler = get_optimizer_scheduler(config, self.model.parameters())

        self.criterion = nn.CrossEntropyLoss().to(device)

    def step(self, iteration):
        self.model.train()
        self.encoder_k.train()

        # data loading
        pos, views_a, views_b = next(self.iterator)
        X1, X2 = views_a.to(self.device, non_blocking=True), views_b.to(self.device, non_blocking=True)

        # forward and backward pass
        self.optimizer.zero_grad()
        with autocast():
            output, target, y1, y2 = self.forward(im_q=X1, im_k=X2)
            loss = self.criterion(output, target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.rank == 0 and self.step_count % self.log_frequency == 0:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(40 * '*')
            print(f"SSL {self.step_count} Loss {loss.item():.3f} ")
            self.writer.ssl_step(iteration, X1, X2, y1, y2, output[:self.batch_size], output[self.batch_size:], loss.item(), self.model.parameters(), lr)

        self.step_count += 1

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        y_q, q = self.model(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            y_k, k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, y_q, y_k


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
