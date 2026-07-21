import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: rag
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # Command: summarize
    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform summarization of results"
    )
    summarize_parser.add_argument("query", type=str, help="Summarize result for the given query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results to summarize")

    # Command: citations
    citations_parser = subparsers.add_parser(
        "citations", help="Retrieve citations for a given query"
    )
    citations_parser.add_argument("query", type=str, help="Query to retrieve citations for")
    citations_parser.add_argument("--limit", type=int, default=5, help="Limit the number of citations to retrieve")

    # Command: question
    question_parser = subparsers.add_parser(
        "question", help="Ask a question based on retrieved documents"
    )
    question_parser.add_argument("query", type=str, help="Question to ask")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit the number of documents to consider")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            search_results, answer = rag_command(query)
            print("Search Results:")
            for result in search_results:
                print(f"- {result['document']['title']}")
            print("\nRAG Response:")
            print(answer)
        case "summarize":
            search_results, answer = summarize_command(args.query, limit=args.limit)
            print("Search Results:")
            for result in search_results:
                print(f"- {result['document']['title']}")
            print("\nLLM Summary:")
            print(answer)
        case "citations":
            search_results, answer = citations_command(args.query, limit=args.limit)
            print("Search Results:")
            for result in search_results:
                print(f"- {result['document']['title']}")
            print("\nLLM Answer:")
            print(answer)
        case "question":
            search_results, answer = question_command(args.query, limit=args.limit)
            print("Search Results:")
            for result in search_results:
                print(f"- {result['document']['title']}")
            print("\nLLM Answer:")
            print(answer)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()