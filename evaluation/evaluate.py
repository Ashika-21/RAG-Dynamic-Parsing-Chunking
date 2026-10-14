def evaluate_retrieval(search_function, index, chunks, test_cases, top_k=3):
    """
    Evaluate retrieval using Hit Rate and MRR.
    """

    total = len(test_cases)
    hits = 0
    reciprocal_ranks = []

    for test in test_cases:
        question = test["question"]
        expected_answer = test["expected_answer"].lower()

        results = search_function(
            question,
            index,
            chunks,
            top_k=top_k
        )

        correct_rank = 0

        for rank, result in enumerate(results, start=1):
            retrieved_text = result["text"].lower()

            if expected_answer in retrieved_text:
                correct_rank = rank
                break

        if correct_rank > 0:
            hits += 1
            reciprocal_rank = 1 / correct_rank
        else:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

        print("\nQuestion:", question)
        print("Expected answer:", test["expected_answer"])

        if correct_rank > 0:
            print("Correct result rank:", correct_rank)
        else:
            print("Correct result rank: Not found")

        print("Reciprocal Rank:", round(reciprocal_rank, 3))

    hit_rate = hits / total if total else 0

    mrr = (
        sum(reciprocal_ranks) / total
        if total
        else 0
    )

    print("\n--- Evaluation Summary ---")
    print("Total questions:", total)
    print("Correctly retrieved:", hits)
    print("Hit Rate:", round(hit_rate, 3))
    print("MRR:", round(mrr, 3))

    return {
        "hit_rate": hit_rate,
        "mrr": mrr
    }