from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch import pca_lowrank, matmul, transpose, swapaxes, tensor, is_tensor, save
from torchvision.utils import make_grid
import numpy as np
import os
from datetime import datetime
from shutil import copy
from math import sqrt
import yaml


def get_gradient_norm(parameters):
    """
    Computes the total l2 norm of gradients
    :param parameters: PyTorch model.parameters()
    :return:
    """
    total_norm = 0
    for p in parameters:
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def get_pca(predictions, project_n_dims=3):
    """
    Computes first project_n_dims Principal Components of a (Batch of) embeddings
    :param predictions: (BS, N, D, H, W) or (N, D, H, W) tensor
    :param project_n_dims: int
    :return: (BS, project_n_dims, D, H, W) or (project_n_dims, D, H, W) tensor
    """
    # convert to torch tensor if input is not already a tensor
    if not is_tensor(predictions):
        pred = tensor(predictions)
    else:
        pred = predictions.clone().detach().float()

    # batched PCA
    if len(pred.shape) == 5:
        batch_size, dims, A, B, C = pred.shape
        embeddings = swapaxes(pred, 0, 1).reshape(dims, -1)
        try:
            U, S, V = pca_lowrank(embeddings)
        except:
            print(embeddings)
            print(embeddings.min(), embeddings.max(), embeddings.mean(), embeddings.std())
        embeddings = matmul(transpose(U[:, :project_n_dims], 0, 1), embeddings)
        return transpose(embeddings.view(project_n_dims, batch_size, A, B, C), 0, 1)

    # Single Image PCA
    elif len(pred.shape) == 4:
        dims, A, B, C = pred.shape
        embeddings = pred.reshape(dims, -1)
        U, S, V = pca_lowrank(embeddings)
        embeddings = matmul(transpose(U[:, :project_n_dims], 0, 1), embeddings)
        return embeddings.view(project_n_dims, A, B, C)

    else:
        raise NotImplementedError('Images have to be 3D')


def get_grid(images, index, mean=None, std=None):
    """
    Computes a grid of images based on a list of slices (index)
    :param images: (tensor) (BS, N, D, H, W)
    :param index: list of integers defining the slices
    :param mean: (float) mean of the dataset's pixelwise intensities
    :param std: (float) std of the dataset's pixelwise intensities
    :return:
    """
    grid = [images[i, :, index[i]] for i in range(len(index))]

    # set range for visualization
    r = None
    if mean and std:
        r = (-mean / std, (255. - mean) / std)
    grid = make_grid(grid,
                     nrow=len(images),
                     normalize=True,
                     range=r,
                     padding=8)
    return grid


class Writer:
    """
    Logging the training process
    """
    def __init__(self, config):
        # create and initialize the log directory
        time = datetime.now()
        tag = os.path.basename(config['tag']).replace('.yaml', '-') if config['tag'] else ''
        dirname = tag + time.strftime('%y-%m-%d-%H:%M:%S') \
            if not os.environ['SLURM_JOB_ID'] \
            else tag + os.environ['SLURM_JOB_ID']

        self.log_dir = os.path.join(config['LOG']['OUTPUT_PATH'], dirname)
        self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')

        if not os.path.isdir(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        print(f'logging training at {self.log_dir}')

        # Dataset details for visualization of images
        self.mean = config['DATASET']['MEAN']
        self.std = config['DATASET']['STD']

        self.writer = SummaryWriter(self.tensorboard_dir)

        # Create a copy of the run config
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def step(self, i, images, target_masks, embedding_pred, mask_pred,  criterion, model_params, lr):
        batch_size, n_channels, x, y, z = images.shape
        size = batch_size if batch_size < 8 else 8
        index = np.random.randint(low=0, high=x, size=size)

        # create grid of images
        grid = get_grid(images, index, mean=self.mean, std=self.std)

        # visualize predictions as PCA images
        pca_features = get_pca(embedding_pred)
        grid_pred = get_grid(pca_features, index)

        # visualize boundary masks
        grid_mask = get_grid(mask_pred, index)
        grid_mask_gt = get_grid(target_masks, index)

        images = make_grid([grid_mask, grid_mask_gt, grid, grid_pred], nrow=1, padding=0, normalize=False)

        # signal amplification
        X_norm = images.norm(2).item()
        out_norm = embedding_pred.norm(2).item()

        # write to log file
        optimizer_log = criterion.get_log()
        train_loss = optimizer_log['total_loss'].mean()
        grad_norm = get_gradient_norm(model_params)
        self.writer.add_image('Inputs, Masks, Boundary, Embedding', images, global_step=i)
        self.writer.add_scalar('Loss/Total', train_loss, global_step=i)
        self.writer.add_scalars('Loss/Distribution',
                                tag_scalar_dict={k: v.mean() for k, v in optimizer_log.items()},
                                global_step=i)
        self.writer.add_scalar('Monitoring/Learning Rate', lr, global_step=i)
        self.writer.add_scalar('Monitoring/Gradients', grad_norm, global_step=i)
        self.writer.add_scalar('Monitoring/Amplification', out_norm / X_norm, global_step=i)

    def ssl_step(self, i, images_a, images_b, y1, y2, z1, z2, loss, model_params, lr):
        batch_size, n_channels, x, y, z = images_a.shape
        size = batch_size if batch_size < 8 else 8
        index = np.random.randint(low=0, high=x, size=size)

        # create grid of images
        grid_images_a = get_grid(images_a[:size], index, mean=self.mean, std=self.std)
        grid_images_b = get_grid(images_b[:size], index, mean=self.mean, std=self.std)

        # visualize predictions as PCA images
        pca = get_pca(torch.cat([y1[:size], y2[:size]]))
        if pca.shape[-3:] != images_a.shape[-3:]:
            pca = F.interpolate(pca, size=images_a.shape[-3:])

        grid_embedding_a = get_grid(pca, index)
        grid_embedding_b = get_grid(pca, index)

        images = make_grid([grid_embedding_a, grid_images_a, grid_images_b, grid_embedding_b],
                           nrow=1, padding=0, normalize=False)


        # signal amplification
        X_norm = torch.mean(torch.norm(torch.cat([images_a, images_b]), dim=1)).item()
        out_norm = torch.mean(torch.norm(torch.cat([y1, y2]), dim=1)).item() / y1.shape[1]

        # collapse metric
        zcat = torch.cat([z1, z2])
        std = torch.mean(torch.std(zcat / torch.norm(zcat, dim=1, keepdim=True), dim=0))

        # write to log file
        self.writer.add_image('Views and Embeddings', images, global_step=i)
        self.writer.add_scalar('Loss/Self-Supervised Loss', loss, global_step=i)
        self.writer.add_scalars('SSL Monitoring/Collapse',
                                tag_scalar_dict={'std(z/|z|)': std.item(),
                                                 '1/sqrt(d)': 1 / sqrt(z1.shape[1])},
                                global_step=i)
        self.writer.add_scalar('SSL Monitoring/Learning Rate', lr, global_step=i)
        self.writer.add_scalar('SSL Monitoring/Gradients', get_gradient_norm(model_params), global_step=i)
        self.writer.add_scalar('SSL Monitoring/Amplification', out_norm / X_norm, global_step=i)
        self.writer.add_scalar('SSL Monitoring/Projection Norm', torch.mean(torch.norm(z1, dim=1)).item(), global_step=i)

    def validate(self, i, loss):
        self.writer.add_scalar('Loss/Validation', loss, global_step=i)

    def save_model(self, i, model, suffix=None):
        model_save_path = os.path.join(self.checkpoint_dir, f'checkpoint_{i}.cpt')
        if not os.path.isdir(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        save(model.state_dict(), model_save_path)

        if suffix:
            copy(model_save_path, os.path.join(os.path.dirname(model_save_path), f'{suffix}_{i}.cpt'))

    def print_model(self, model, images):
        self.writer.add_graph(model, images, verbose=False)
