import torch
from torch.nn import Module, BCEWithLogitsLoss


def get_embeddings(pred, target, target_masks, idx):
    mask_idx = target == idx
    mask = mask_idx & target_masks.bool()
    mask = torch.swapaxes(mask, 0, 1)
    pred = torch.swapaxes(pred, 0, 1)
    embeddings_idx = pred[..., mask.squeeze()]
    return embeddings_idx


def get_individual_embeddings(pred, target, target_masks, idx):
    mask_idx = target == idx
    mask = mask_idx & target_masks.bool()
    embeddings_idx = pred[..., mask.squeeze()]
    return embeddings_idx


class EmbeddingLoss(Module):
    def __init__(self, alpha: float, beta: float, gamma: float, delta: float, device: torch.device, average=16):
        super(EmbeddingLoss, self).__init__()
        self.device = device
        self.background_module = BCEWithLogitsLoss()

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

        self.internal = torch.Tensor([-1])
        self.external = torch.Tensor([-1])
        self.regularization = torch.Tensor([-1])
        self.background = torch.Tensor([-1])
        self.total_loss = torch.Tensor([-1])

        # monitoring
        self.average = average
        self._log = {
            'internal': torch.zeros(average),
            'external': torch.zeros(average),
            'regularization': torch.zeros(average),
            'background': torch.zeros(average),
            'total_loss': torch.zeros(average)
        }
        self.permute = torch.zeros(self.average, dtype=torch.int64)
        self.permute[:-1] = torch.arange(self.average, dtype=torch.int64)[1:]

    def forward(self, embeddings: torch.Tensor, masks: torch.Tensor, target_embeddings: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        self.background = self.background_module(masks, target_masks)
        self._faster_embedding_loss(embeddings, target_embeddings, target_masks)
        self.total_loss = self.alpha * self.internal + \
                          self.beta * self.external + \
                          self.gamma + self.regularization + \
                          self.background

        # monitoring
        self.log()

        return self.total_loss

    def log(self):
        for k in self._log.keys():
            self._log[k] = self._log[k][self.permute]
            self._log[k][-1] = getattr(self, k).item()

    def get_log(self):
        return self._log

    def _faster_embedding_loss(self, embeddings: torch.Tensor,
                               target: torch.Tensor,
                               target_masks: torch.Tensor):
        self.internal, self.external, self.regularization = faster_embedding_loss(embeddings,
                                                                                  target,
                                                                                  target_masks,
                                                                                  self.device,
                                                                                  self.delta)


def faster_embedding_loss(embeddings: torch.Tensor,
                          target: torch.Tensor,
                          target_masks: torch.Tensor,
                          device: torch.cuda.device,
                          delta: float):
    batch_size = embeddings.shape[0]
    internal = torch.zeros(batch_size, device=device)
    external = torch.zeros(batch_size, device=device)
    regularization = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        target_i = target[i]
        mask_i = target_masks[i]
        embeddings_i = embeddings[i]

        clusters = torch.unique(target_i)

        foreground = torch.ones(len(clusters), dtype=torch.bool, requires_grad=False, device=device)
        cluster_centers = torch.zeros((len(clusters), embeddings.shape[1]), device=device)
        internal_loss = torch.zeros_like(clusters, device=device).float()

        for j, cluster in enumerate(clusters):
            # get a list of embedding vectors for the corresponding cluster
            embeddings_c = get_individual_embeddings(embeddings_i, target_i, mask_i, cluster)
            if embeddings_c.nelement() == 0 or cluster == 0:
                foreground[j] = 0
                continue

            # compute centers of these clusters in embeddings space
            center = embeddings_c.mean(dim=1)
            cluster_centers[j] = center

            # compute the internal loss
            internal_loss[j] = torch.mean(torch.norm(embeddings_c - center.view(-1, 1), dim=0)**2)

        # compute external loss
        C = torch.sum(foreground)
        cluster_centers = cluster_centers[foreground]

        if C >= 1:
            internal[i] = internal_loss[foreground].mean()
            regularization[i] = torch.mean(torch.norm(cluster_centers, dim=1))

        if C >= 2:
            # compute pairwise distances between clusters
            ind = torch.arange(C, device=device).view(-1, 1).repeat(1, C).flatten()
            center_distances = torch.norm(cluster_centers.repeat(C, 1) - cluster_centers[ind], dim=1).view(C, C)
            center_distances = torch.maximum(2 * delta - center_distances, torch.zeros((C, C), device=device)) ** 2

            # set diagonal to zero
            eye_mask = torch.eye(C, C, device=device, requires_grad=False).bool()
            center_distances.masked_fill_(eye_mask, 0)

            # save external loss results
            external[i] = center_distances.sum() / C / (C - 1)

    internal = internal.mean()
    external = external.mean()
    regularization = regularization.mean()
    return internal, external, regularization
