import pickle


class Tokenizer:
    def __init__(self, corpus) -> None:
        self.stoi = {}
        self.itos = {}
        self.vocab = set()
        self.vocab_size = 0

        self._create_vocab(corpus)

    def _create_vocab(self, corpus):
        wordlist = []

        for doc in corpus:
            wordlist.extend(self._split_text(doc["text"].lower()))

        self.vocab = sorted(set(wordlist))
        self.stoi = {word: idx for (idx, word) in enumerate(self.vocab)}
        self.itos = {idx: word for (word, idx) in self.stoi.items()}
        self.vocab_size = len(self.vocab)

    def _split_text(self, text):
        return text.split(" ")

    def encode(self, text):
        return [self.stoi[word] for word in self._split_text(text.lower())]

    def decode(self, tokens):
        words = [self.itos[idx] for idx in tokens]
        return " ".join(words)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
