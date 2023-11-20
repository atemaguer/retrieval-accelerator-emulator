import torch
from indexer import RoutingNetwork

corpus = [
    [0, torch.tensor([1, 0, 1, 0, 0])],
    [1, torch.tensor([1, 0, 0, 1, 1])],
    [2, torch.tensor([0, 0, 1, 1, 1])],
    [3, torch.tensor([1, 1, 1, 0, 0])],
    [4, torch.tensor([0, 1, 0, 0, 1])],
]


def test_routing_network_indexing():
    indexer = RoutingNetwork(5, 5, [0, 1, 2, 3, 4])
    indexer.index(corpus)
    expected = corpus[0][1]
    output = indexer.get_scoring_unit_input(corpus[0][0])
    assert torch.equal(expected, output), "scoring units not configured correctly."


def test_routing_network_scoring():
    query = torch.tensor([1, 1, 0, 1, 0])
    indexer = RoutingNetwork(5, 5, [0, 1, 2, 3, 4])
    indexer.index(corpus)
    assert indexer.get_scoring_unit_score(query, 0) == 1, "scores don't match."
    assert indexer.get_scoring_unit_score(query, 3) == 2, "scores don't match."
