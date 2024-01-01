import torch
from typing import Iterable, Dict
from torch import nn, Tensor
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


def get_hilbert_distance(X, temperature):
    funk = X.matmul((1 / X).T)
    funk_ab = (funk.tril(diagonal=-1)).T
    funk_ba = funk.triu(diagonal=1)

    distance_matrix = funk_ab.mul(funk_ba)
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix = distance_matrix.log() / temperature
    return distance_matrix


class SimcseHilbertLoss(nn.Module):
    def __init__(self, device, backbone: SentenceTransformer, simcse_temperature, hilbert_temperature, lambda_value,
                 embedding_size):
        super(SimcseHilbertLoss, self).__init__()
        self.device = device
        self.model = backbone
        self.simcse_temperature = simcse_temperature
        self.hilbert_temperature = hilbert_temperature
        self.lambda_value = lambda_value
        self.embedding_size = embedding_size
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax_function = nn.Softmax(1)
        self.projection = AddProjection(model=self.model, embedding_size=self.embedding_size)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = reps[1]
        batch_size = embeddings_a.shape[0]

        ## simcse loss
        LARGE = 1e9
        embeddings_ab = torch.cat([embeddings_a, embeddings_b])
        sim_matrix = psim(embeddings_ab, embeddings_ab) / self.simcse_temperature

        # set logit for a label with itself to a large negative number
        sim_matrix = sim_matrix - LARGE * torch.eye(sim_matrix.shape[0], device=self.device)
        labels = torch.tensor(
            list(range(batch_size, 2 * batch_size)) + list(range(batch_size)),
            device=self.device,
        )
        simcse_loss = self.lambda_value * self.cross_entropy(sim_matrix, labels)

        ## hilbert loss
        z1, z2 = self.projection(embeddings_a), self.projection(embeddings_b)
        softmax1, softmax2 = self.softmax_function(z1), self.softmax_function(z2)
        embeddings_z = torch.cat([softmax1, softmax2])

        dist_matrix = get_hilbert_distance(embeddings_z, self.hilbert_temperature)
        sim_matrix = 1 / (dist_matrix + 1)
        sim_matrix = sim_matrix - LARGE * torch.eye(sim_matrix.shape[0], device=self.device)
        hilbert_loss = self.cross_entropy(sim_matrix, labels)

        loss = simcse_loss + hilbert_loss
        return loss