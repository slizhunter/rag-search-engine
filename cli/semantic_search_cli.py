import argparse

from lib.search_utils import load_movies
from lib.semantic_search import SemanticSearch, embed_query_text, embed_text, verify_embeddings, verify_model, semantic_search, chunk_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: verify
    subparsers.add_parser(
        "verify", help="Verify that the semantic search model loads correctly"
    )

    # Command: embed_text
    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for a given text"
    )
    embed_parser.add_argument("text", type=str, help="Text to generate embeddings for")

    # Command: verify_embeddings
    subparsers.add_parser(
        "verify_embeddings", help="Verify that the embeddings are generated correctly"
    )

    # Command: embed_query
    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate embeddings for a given query text"
    )
    embed_query_parser.add_argument("text", type=str, help="Query text to generate embeddings for")

    # Command: search
    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    # Command: chunk
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk text into smaller pieces"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="Size of each chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="Number of overlapping words between chunks"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()