import pandas as pd

from parsing.parser import parse_pdf, validate_parsing
from chunking.chunker import (
    fixed_chunking,
    structural_chunking,
    dynamic_chunking
)
from retrieval.retriever import (
    create_vector_store,
    search_chunks
)
from evaluation.evaluate import evaluate_retrieval


PDF_PATH = "data/sample.pdf"

TEST_CASES = [
    {
        "question": "How many casual leave days are provided per year?",
        "expected_answer": "12 days of casual leave"
    },
    {
        "question": "How should an employee report a missing attendance record?",
        "expected_answer": "inform the HR team within two working days"
    },
    {
        "question": "What is the monthly internet support allowance?",
        "expected_answer": "500"
    }
]


def main():
    # Step 1: Parse the PDF
    print("Starting PDF Parsing...\n")

    parsed_blocks = parse_pdf(PDF_PATH)
    result = validate_parsing(parsed_blocks)

    print("Parsing Status:", result)

    if not result["success"]:
        return

    # Step 2: Apply different chunking methods
    chunking_methods = {
        "Fixed Chunking": fixed_chunking(parsed_blocks),
        "Structural Chunking": structural_chunking(parsed_blocks),
        "Dynamic Chunking": dynamic_chunking(parsed_blocks)
    }

    print("\n========== CHUNKING SUMMARY ==========")

    for name, chunks in chunking_methods.items():
        print(f"{name}: {len(chunks)} chunks")

    # Step 3: Evaluate retrieval for each method
    evaluation_results = {}

    print("\n========== RETRIEVAL EVALUATION ==========")

    for name, chunks in chunking_methods.items():
        print(f"\n===== {name} =====")

        index, stored_chunks = create_vector_store(chunks)

        metrics = evaluate_retrieval(
            search_function=search_chunks,
            index=index,
            chunks=stored_chunks,
            test_cases=TEST_CASES,
            top_k=3
        )

        evaluation_results[name] = metrics

    # Step 4: Compare the results
    print("\n========== FINAL COMPARISON ==========")

    comparison_data = []

    for name, chunks in chunking_methods.items():
        metrics = evaluation_results[name]

        comparison_data.append({
            "Chunking Method": name,
            "Number of Chunks": len(chunks),
            "Hit Rate": metrics["hit_rate"],
            "MRR": metrics["mrr"]
        })

    comparison_df = pd.DataFrame(comparison_data)

    print(comparison_df.to_string(index=False))

    # Step 5: Save the comparison as a CSV file
    output_path = "evaluation_results.csv"

    comparison_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()