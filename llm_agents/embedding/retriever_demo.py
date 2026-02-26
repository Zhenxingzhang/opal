"""Example: index documents and query them using SemanticRetriever."""

from llm_agents.embedding import SemanticRetriever

# Example documents to index
DOCS = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Python is a popular programming language.",
    "Machine learning models can classify text.",
    "The weather today is sunny and warm.",
    "Neural networks are inspired by the human brain.",
    "I enjoy hiking in the mountains on weekends.",
    "Financial markets experienced high volatility this quarter.",
]


def main() -> None:
    retriever = SemanticRetriever()
    retriever.index(DOCS)
    print(f"Indexed {retriever.document_count} documents")

    queries = [
        "What animals make good pets?",
        "Tell me about artificial intelligence.",
        "How is the stock market doing?",
    ]

    for query in queries:
        results = retriever.search(query, top_k=3)
        print(f"\nQuery: {query!r}")
        for rank, result in enumerate(results, 1):
            print(f"  {rank}. [{result.score:.4f}] {result.text}")


if __name__ == "__main__":
    main()
