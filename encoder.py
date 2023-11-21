import torch
from torch import nn

class BoWEncoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, vocab=None) -> None:
        super().__init__()

        self.bow_model = nn.Embedding(vocab_size, embed_dim)
        self.bow_model.weight.data.copy_(torch.eye(embed_dim))
        self.bow_model.weight.requires_grad = False
        self.vocab = vocab

    def forward(self, x):
        one_hot_embeddings = self.bow_model(x)
        return one_hot_embeddings.sum(0)
