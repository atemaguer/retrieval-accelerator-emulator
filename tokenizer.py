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
        tokens = []
        for word in self._split_text(text.lower()):
            if word in self.stoi:
                tokens.append(self.stoi[word])
        return tokens

    def decode(self, tokens):
        wordlist = [self.itos[idx] for idx in tokens]
        return " ".join(wordlist)

    def save_state(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
