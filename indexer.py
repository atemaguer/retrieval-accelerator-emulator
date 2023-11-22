import torch

from dataclasses import dataclass

@dataclass
class ScoringUnit:
    score: int
    id: int

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
            self.network[idx, :] = doc["embeddings"].squeeze(0)
            counter += 1

            if counter % 50 == 0:
                print(f"{counter} documents indexed.")

    def get_scoring_unit_input(self, pid):
        idx = self.pid_to_idx[pid]
        return self.network[idx, :]

    def get_scoring_unit_score(self, q_vector, pid):
        idx = self.pid_to_idx[pid]
        all_scores = self.network.bool() & q_vector.bool()
        return all_scores[idx, :].sum()

    def get_query_tf_idf_score(self, q_vector):
        terms_sum = self.network.sum(-1)

        tf = self.network[:, q_vector.bool()] / terms_sum.unsqueeze(1)

        mask = self.network.bool() & q_vector.bool()

        num_docs_with_term = mask[:, q_vector.bool()].int().sum(0)

        idf = self.d_size * (1 / num_docs_with_term)

        tf_idf = tf * idf

        tf_idf_scores = tf_idf.sum(-1)

        pid_scores = [
            ScoringUnit(id=self.idx_to_pid[i], score=score.item()) for i, score in enumerate(tf_idf_scores)
        ]

        return pid_scores
    
    def get_query_score(self, q_vector):
        mask = q_vector.bool()
        scores = (self.network.bool() & mask).int().sum(dim=-1)

        pid_scores = [
            ScoringUnit(id=self.idx_to_pid[i], score=score.item()) for i, score in enumerate(scores)
        ]
        
        return pid_scores

    def get_query_energy_consumption(self, q_vector):
        mask = q_vector.bool()

        states = (self.network.bool() & mask)
        num_active_input_wires = mask.int().sum()
        num_active_output_wires = states.int().sum()

        return self.d_size * num_active_input_wires + num_active_output_wires * self.v_size

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
