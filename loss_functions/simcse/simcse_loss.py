import torch
import torch.nn as nn
from typing import Iterable, Dict
from torch import Tensor
from sentence_transformers import SentenceTransformer


class AddProjection(nn.Module):
    def __init__(self, model: SentenceTransformer, embedding_size):
        super(AddProjection, self).__init__()
        self.model = model
        self.embedding_size = embedding_size
        self.mlp_dim = self.model.get_sentence_embedding_dimension()

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mlp_dim, out_features=self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
        )

    def forward(self, a: Tensor):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        return self.projection(a)


def psim(X, Y):
    """ Computes all the pairwise similarities

    Parameters
    ----------
    X : torch.tensor
        shape [n, d]
    Y : torch.tensor
        shape [m, d]

    Returns
    -------
    torch.tensor
        shape [n, m] of all pairwise similarities
    """
    SMALL = 1e-12
    n, m = X.shape[0], Y.shape[0]
    X_norm = ((X ** 2).sum(1) + SMALL).sqrt()
    Y_norm = ((Y ** 2).sum(1) + SMALL).sqrt()
    X_dot_Y = X @ Y.T

    ret = X_dot_Y / (
            (X_norm.unsqueeze(1) @ torch.ones((1, m), device=X.device))
            * (torch.ones((n, 1), device=Y.device) @ Y_norm.unsqueeze(0))
    )
    return ret


class SimCSELoss(nn.Module):
    def __init__(self, temperature, backbone: SentenceTransformer, device, embedding_size, lambda_value):
        super().__init__()
        self.temperature = temperature
        self.model = backbone
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.lambda_value = lambda_value
        self.projection = AddProjection(model=self.model, embedding_size=self.embedding_size)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in
                sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        embeddings_a, embeddings_b = self.projection(embeddings_a), self.projection(embeddings_b)

        LARGE = 1e9
        batch_size = embeddings_a.shape[0]
        embeddings_ab = torch.cat([embeddings_a, embeddings_b])
        sim = psim(embeddings_ab, embeddings_ab) / self.temperature

        # set logit for a label with itself to a large negative number
        sim = sim - LARGE * torch.eye(sim.shape[0], device=self.device)
        labels = torch.tensor(
            list(range(batch_size, 2 * batch_size)) + list(range(batch_size)),
            device=self.device,
        )
        loss = self.lambda_value * self.cross_entropy(sim, labels)
        return loss