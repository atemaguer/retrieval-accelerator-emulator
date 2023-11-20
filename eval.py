def evaluate(retrieval_accelerator, documents, queries, relevance_labels):
    # Indexing phase
    for pid, document in documents.items():
        encoded_document = retrieval_accelerator.encoder.encode(document)
        retrieval_accelerator.routing_network.memorize(encoded_document, pid)

    # Retrieval phase
    for qid, query in queries.items():
        encoded_query = retrieval_accelerator.encoder.encode(query)
        retrieval_accelerator.routing_network.memorize(encoded_query, qid)

        # Update scores in scoring units
        for pid, document in documents.items():
            for word in encoded_query:
                if (
                    retrieval_accelerator.routing_network.network[pid][
                        retrieval_accelerator.encoder.vocabulary.index(word)
                    ]
                    == 1
                ):
                    retrieval_accelerator.scoring_units[pid].update_score(1)

        # Trigger ranking
        top_scoring_units = retrieval_accelerator.ranking_network.rank(
            retrieval_accelerator.scoring_units
        )

        # Compute Recall@1000
        relevant_documents = set(relevance_labels[qid])
        retrieved_documents = set([unit.pid for unit in top_scoring_units])
        recall = len(relevant_documents & retrieved_documents) / len(relevant_documents)

        print(f"Recall@1000 for query {qid}: {recall}")
