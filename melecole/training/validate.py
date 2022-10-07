import torch
from torch.cuda.amp import autocast
import GPUtil
from loss.embedding_loss import EmbeddingLoss


class Validator:
    def __init__(self, dataloader, writer, device, config, verbose=True):
        self.dataloader = dataloader
        self.delta = config['LOSS']['DELTA']
        self.criterion = EmbeddingLoss(alpha=config['LOSS']['ALPHA'],
                                       beta=config['LOSS']['BETA'],
                                       gamma=config['LOSS']['GAMMA'],
                                       delta=config['LOSS']['DELTA'],
                                       device=device,
                                       average=1)

        self.writer = writer
        self.supervised_validation_frequency = -1 if 'SUPERVISED_VALIDATION' not in config['LOG'].keys() \
            else config['LOG']['SUPERVISED_VALIDATION']
        self.ssl_validation_frequency = -1 if 'UNSUPERVISED_VALIDATION' not in config['LOG'].keys() \
            else config['LOG']['UNSUPERVISED_VALIDATION']

        self.device = device

        self.verbose = verbose
        self.initial_print_gpu = True

    def validate(self,
                 iteration,
                 model
                 ):
        """
        Run validation a cuda device (rank). Today (02/2022) distributed validation does not seem to be possible.
        Thus, this function may only be executed by one process.
        :param rank:
        :param dataloader:
        :param model:
        :param criterion:
        :param device:
        :param delta:
        :param validate_on_targets:
        :return:
        """
        loss = 0.0

        model.eval()

        for i, (pos, volume, target, _) in enumerate(self.dataloader):

            with torch.no_grad():
                X = volume.to(self.device, non_blocking=True)
                y_instances = target[0].to(self.device)
                y_foreground_mask = target[1].to(self.device)

                with autocast():
                    output = model(X)
                    loss += self.criterion(embeddings=output.embeddings,
                                           masks=output.boundaries,
                                           target_embeddings=y_instances,
                                           target_masks=y_foreground_mask).item()

        loss = loss / (len(self.dataloader) + 1)

        if self.verbose:
            if self.initial_print_gpu:
                GPUtil.showUtilization(all=True)
                self.initial_print_gpu = False

            print(40 * '*')
            print(f"VALIDATION  {iteration} // {loss:.3e}")
            print(40 * '*')

        self.writer.validate(iteration,
                             loss)
