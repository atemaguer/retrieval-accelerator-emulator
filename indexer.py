import torch
from torch import nn


class ScoringUnit:
    def __init__(self) -> None:
        pass

    def update(self, signal):
        pass

    def get_score(self):
        pass


class BinaryScoringUnit(ScoringUnit):
    pass


class IntergerScoringUnit(ScoringUnit):
    pass


class RoutingNetwork:
    def __init__(self, v_size, d_size, pids) -> None:
        super().__init__()
        self.v_size = v_size
        self.d_size = d_size
        self.network = torch.zeros((d_size, v_size))
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
        self.idx_to_pid = {idx: pid for pid, idx in self.pid_to_idx.items()}

    def index(self, documents) -> None:
        counter = 0

        for doc in documents:
            idx = self.pid_to_idx[doc["pid"]]
            self.network[idx, :] += doc["embeddings"].squeeze(0)
            counter += 1

            print(f"{counter} documents indexed.")

    def get_scoring_unit_input(self, pid):
        idx = self.pid_to_idx[pid]
        return self.network[idx, :]

    def get_scoring_unit_score(self, q_vector, pid):
        idx = self.pid_to_idx[pid]
        scores = self.network[idx, :].bool() & q_vector.bool()
        return scores.sum()

    def batch_score_query(self, q_vector):
        pass

    def score_query(self, q_vector):
        q_vector_bool = q_vector.bool()
        scores = (self.network.bool() & q_vector_bool).sum(dim=-1)
        pid_scores = [
            (self.idx_to_pid[i], score.item()) for i, score in enumerate(scores)
        ]
        return pid_scores

    def save_state(self, file_path):
        state = {
            "v_size": self.v_size,
            "d_size": self.d_size,
            "network": self.network,
            "pid_to_idx": self.pid_to_idx,
            "idx_to_pid": self.idx_to_pid,
        }
        torch.save(state, file_path)

    @classmethod
    def load_state(cls, file_path):
        state = torch.load(file_path)
        routing_network = cls(
            state["v_size"], state["d_size"], list(state["pid_to_idx"].keys())
        )
        routing_network.network = state["network"]
        routing_network.pid_to_idx = state["pid_to_idx"]
        routing_network.idx_to_pid = state["idx_to_pid"]
        return routing_network
