# RAG Dynamic Parsing and Dynamic Chunking

## Project Overview

This project implements a simple PDF ingestion and retrieval pipeline. It extracts text from a PDF, divides the text into chunks using different methods, creates embeddings, and retrieves relevant chunks based on user questions.

## Objectives

- Extract text from PDF documents.
- Compare Fixed, Structural, and Dynamic Chunking.
- Generate text embeddings.
- Retrieve relevant chunks using FAISS.
- Evaluate retrieval performance using Hit Rate and Mean Reciprocal Rank (MRR).

## Technologies Used

- Python
- PyMuPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Pandas
- NumPy

## Project Structure

- `data/` – Contains the sample PDF.
- `parsing/` – Extracts text from the PDF.
- `chunking/` – Implements three chunking methods.
- `retrieval/` – Creates embeddings and retrieves relevant chunks.
- `evaluation/` – Evaluates retrieval performance.
- `main.py` – Runs the complete pipeline.
- `requirements.txt` – Lists the required packages.
- `evaluation_results.csv` – Stores the comparison results.

## Chunking Methods

1. **Fixed Chunking:** Divides text into fixed-size chunks with overlap.
2. **Structural Chunking:** Groups content based on document headings.
3. **Dynamic Chunking:** Groups content based on a maximum chunk-size limit.

## Evaluation Metrics

- **Hit Rate:** Measures whether a relevant result appears in the top retrieved results.
- **MRR:** Measures how highly the first relevant result is ranked.

## How to Run

1. Install the required packages:

   `pip install -r requirements.txt`

2. Run the project:

   `python main.py`

3. Check `evaluation_results.csv` for the comparison results.

## Result

In the current test with three questions, all three chunking methods achieved a Hit Rate of 1.0. Structural Chunking and Dynamic Chunking achieved an MRR of 1.0, while Fixed Chunking achieved an MRR of 0.5.

These results are based on a small test set and may change with more questions.