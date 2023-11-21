import torch
import json

from retriever import Retriever, Ranker
from encoder import BoWEncoder
from indexer import RoutingNetwork
from tokenizer import Tokenizer

from utils import get_dataset
from eval import evaluate


queries_path = "./dev_queries/raw.tsv"
queries = get_dataset(queries_path)

labels_path = "./dev_qrel.json"
labels = json.load(open(labels_path, "r"))

if __name__ == "__main__":
    tokenizer = Tokenizer.load_state("./data/tokenizer.pkl")

    print("tokenizer loaded")

    encoder = BoWEncoder(
        embed_dim=tokenizer.vocab_size, vocab_size=tokenizer.vocab_size
    )

    indexer = RoutingNetwork.load_state("./data/index.pkl")

    print("index loaded")

    ranker = Ranker(r_steps=1000)

    retriever = Retriever(tokenizer, encoder, indexer)

    print("calculating summary statistics...")
    
    evaluate(retriever, ranker, queries, labels)

