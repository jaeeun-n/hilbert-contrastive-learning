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

    def forward(self, a: Tensor, doc: Tensor):
        return self.projection(a), self.projection(doc)


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


def get_hilbert_distance(X, Y, temperature):
    funk_ab = X.matmul(1 / Y.T)
    funk_ba = Y.matmul(1 / X.T)
    dist_matrix = (funk_ab.mul(funk_ba)).log() / temperature
    return dist_matrix


class SimcseHilbertLoss2Branch(nn.Module):
    def __init__(self, device, backbone: SentenceTransformer, simcse_temperature, hilbert_temperature, lambda_value,
                 embedding_size):
        super(SimcseHilbertLoss2Branch, self).__init__()
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
        embeddings_c = reps[2]
        embeddings_doc = reps[3]
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
        projection_c, projection_doc = self.projection(embeddings_c, embeddings_doc)
        softmax_c, softmax_doc = self.softmax_function(projection_c), self.softmax_function(projection_doc)

        dist_matrix = get_hilbert_distance(softmax_c, softmax_doc, self.hilbert_temperature)
        sim_matrix = 1 / (dist_matrix + 1)
        labels = torch.tensor(list(range(batch_size)), device=self.device)
        hilbert_loss = self.cross_entropy(sim_matrix, labels)

        loss = simcse_loss + hilbert_loss
        return loss