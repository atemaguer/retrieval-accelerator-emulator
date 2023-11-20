import torch
import multiprocessing as mp
from functools import partial

from retriever import Retriever
from encoder import BoWEncoder
from indexer import RoutingNetwork
from tokenizer import Tokenizer

from utils import get_dataset

documents_path = "./full_collection/raw.tsv"
documents = get_dataset(documents_path)[:10]
pids = [doc["pid"] for doc in documents]

dev_set_path = "./dev_queries/raw.tsv"
dev_set = get_dataset(dev_set_path)


def tokenize_doc(tokenizer, doc):
    return {"token_ids": torch.tensor(tokenizer.encode(doc["text"])), **doc}


def encode_doc(encoder, doc):
    return {"embeddings": encoder(doc["token_ids"]), **doc}


if __name__ == "__main__":
    # loading a tokenizer that was created from the corpus.
    tokenizer = Tokenizer.load("./data/tokenizer.pkl")

    tokenized_docs = [
        {"token_ids": torch.tensor(tokenizer.encode(doc["text"])), **doc}
        for doc in documents
    ]

    print("done tokenizing")

    encoder = BoWEncoder(
        embed_dim=tokenizer.vocab_size, vocab_size=tokenizer.vocab_size
    )

    doc_embeddings = [
        {"embeddings": encoder(doc["token_ids"]), **doc} for doc in tokenized_docs
    ]

    print("done embedding")

    indexer = RoutingNetwork(tokenizer.vocab_size, len(documents), pids)
    indexer.index(doc_embeddings)

    print("done indexing")
    retriever = Retriever(tokenizer, encoder, indexer)

    scores = retriever.search("George Washington")

    print(len(scores))
