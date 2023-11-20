from tokenizer import Tokenizer
from utils import get_dataset

corpus_path = "./full_collection/raw.tsv"
corpus = get_dataset(corpus_path)

tokenizer = Tokenizer.load("./data/tokenizer.pkl")

print(tokenizer.vocab_size)
