import torch
import heapq

class Ranker:
    def __init__(self, r_steps) -> None:
        self.r_steps = r_steps
        self.heap = []

    def rank(self, scoring_units):
        for unit in scoring_units:
            heapq.heappush(self.heap, (-unit.score, unit.id, unit))

        top_units = [heapq.heappop(self.heap)[2] for _ in range(self.r_steps)]

        return top_units

class Retriever:
    def __init__(self, tokenizer, encoder, indexer) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.indexer = indexer

    def search(self, query):
        q_tokens = torch.tensor(self.tokenizer.encode(query))
        q_vector = self.encoder(q_tokens)
        # scores = self.indexer.get_query_score(q_vector)
        scores = self.indexer.get_query_tf_idf_score(q_vector)
        
        return scores
