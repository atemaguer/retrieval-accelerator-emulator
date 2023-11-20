import torch
import heapq


class RankingNetwork:
    def __init__(self, r_steps) -> None:
        self.r_steps = r_steps
        self.heap = []

    def rank(self, scoring_units):
        for unit in scoring_units:
            if len(self.heap) < self.r_steps:
                heapq.heappush(self.heap, (-unit.score, unit))
            else:
                heapq.heappushpop(self.heap, (-unit.score, unit))

        top_units = [heapq.heappop(self.heap)[1] for _ in range(len(self.heap))]
        top_units.reverse()

        return top_units


class Retriever:
    def __init__(self, tokenizer, encoder, indexer) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.indexer = indexer

    def search(self, query):
        q_tokens = torch.tensor(self.tokenizer.encode(query))
        q_vector = self.encoder(q_tokens)
        scores = self.indexer.score_query(q_vector)
        return scores
