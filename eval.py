import torch
import numpy as np
import matplotlib.pyplot as plt

def _convert_rankings_to_dict(rankings) -> dict:
    rankings_dict = {str(unit.id): 1 for unit in rankings}
    return rankings_dict

def _calculate_recall(labels: dict, rankings: dict) -> float:
    relevant_docs = set(labels.keys())
    retrieved_docs = set(rankings.keys())
    relevant_retrieved_docs = relevant_docs.intersection(retrieved_docs)
    
    if len(relevant_docs) == 0:
        return 0.0
    else:
        return len(relevant_retrieved_docs) / len(relevant_docs)

def _calculate_energy_consumption(retriever, q_text):
    q_tokens = torch.tensor(retriever.tokenizer.encode(q_text))
    q_embeds = retriever.encoder(q_tokens)
    energy_consumed = retriever.indexer.get_query_energy_consumption(q_embeds)
    return energy_consumed

def summary_statistics(values):
    values = np.array(values)
    
    mean = np.mean(values)
    print(f"Mean: {mean}")
    
    std = np.std(values)
    print(f"Standard Deviation: {std}")
    
    p95 = np.percentile(values, 95)
    print(f"P95: {p95}")
    
    # Generate a histogram of the values
    plt.hist(values, bins=20, edgecolor='black')
    plt.title("Histogram Of Energy Consumption")
    plt.xlabel("Energy Value")
    plt.ylabel("Frequency")
    plt.show()

def evaluate(retriever, ranker, queries, labels):
    recalls = []
    energy_vals = []
    for query in queries:

        scores = retriever.search(query["text"])
        rankings = ranker.rank(scores)
        rankings = _convert_rankings_to_dict(rankings)
        recall = _calculate_recall(labels[str(query["pid"])], rankings)
        energy_consumed = _calculate_energy_consumption(retriever, query["text"])
                                                        
        recalls.append(recall)
        energy_vals.append(energy_consumed)

    average_recall = round(sum(recalls) / len(recalls), 3) * 100

    print(f"Recall@1000 is {average_recall} %")

    summary_statistics(energy_vals)
