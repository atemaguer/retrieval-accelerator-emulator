import torch 

from tokenizer import Tokenizer
from indexer import RoutingNetwork
from encoder import BoWEncoder
from utils import get_dataset

corpus_path = "./full_collection/raw.tsv"
documents = get_dataset(corpus_path)
pids = [doc["pid"] for doc in documents]

tokenizer = Tokenizer.load_state("./data/tokenizer.pkl")
indexer = RoutingNetwork(tokenizer.vocab_size, len(documents), pids)
encoder = BoWEncoder(tokenizer.vocab_size, tokenizer.vocab_size)

print("tokenizing...")
tokenized_docs = [
        {"token_ids": torch.tensor(tokenizer.encode(doc["text"])), **doc}
        for doc in documents
    ]
print("tokenizing compelete")
print("embedding documents...")
doc_embeddings = [
        {"embeddings": encoder(doc["token_ids"]), **doc} for doc in tokenized_docs
    ]

print("embedding documents complete...")
print("indexing documents...")

indexer.index(doc_embeddings)
indexer.save_state("./data/index.pkl")

print("indexing complete.")