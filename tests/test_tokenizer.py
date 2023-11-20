from tokenizer import Tokenizer

dataset = [
    {
        "pid": 1,
        "text": "Once upon a time in a town far away, a small boy named Tim lived with his parents. He was always curious about the world around him."
    },
    {
        "pid": 2,
        "text": "In the heart of the city, a renowned chef named Julia was preparing her signature dish. Her food was loved by everyone in town."
    },
    {
        "pid": 3,
        "text": "On a bright sunny day, a group of friends decided to go on a hike. They packed their bags and set off early in the morning."
    },
    {
        "pid": 4,
        "text": "In the quiet town of Bakersville, an annual pie contest was held. The aroma of freshly baked pies filled the air."
    },
    {
        "pid": 5,
        "text": "In the depths of the ocean, a school of fish swam swiftly. They were heading towards the coral reef, their home."
    },
    {
        "pid": 6,
        "text": "In the bustling city of New York, a young artist was painting a mural. The vibrant colors reflected the city's energy."
    },
    {
        "pid": 7,
        "text": "In a small village, an old man was telling stories to the children. His tales were full of wisdom and adventure."
    },
    {
        "pid": 8,
        "text": "On top of a hill, a young girl was flying her kite. The wind was perfect, and the kite soared high in the sky."
    },
    {
        "pid": 9,
        "text": "In a dense forest, a pack of wolves was on the hunt. They moved silently, blending in with the surroundings."
    },
    {
        "pid": 10,
        "text": "In a distant galaxy, a group of astronauts was exploring a new planet. They were in awe of the alien landscape."
    }
]

def test_encode_decode():
    corpus = [sample["text"] for sample in dataset]

    tokenizer = Tokenizer(corpus)
    input_text = "pack of wolves"

    assert tokenizer.decode(tokenizer.encode(input_text)) == input_text, "decoded text doesn't match input."