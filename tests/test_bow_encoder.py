import torch
from encoder import BoWEncoder

vocab_size = 5
model = BoWEncoder(vocab_size, vocab_size)

def test_embeddings():
    expected = torch.tensor([1, 1, 0, 0, 1])
    output = model(torch.tensor([0, 1, 4]))
    print(expected, output)
    assert output.size() == expected.size(), "embeddings don't have similar sizes"
    assert torch.equal(output, expected), "embeddings didn't match"
