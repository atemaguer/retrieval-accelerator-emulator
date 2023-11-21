from retriever import Ranker
from indexer import ScoringUnit

def test_ranking():
    units = [ScoringUnit(0, 1), ScoringUnit(0, 2), ScoringUnit(1, 3)]
    ranker = Ranker(3)
    rankings = ranker.rank(units)
    expected = [ScoringUnit(1, 3), ScoringUnit(0, 2), ScoringUnit(0, 1)]