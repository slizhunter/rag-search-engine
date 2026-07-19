import argparse

from lib.hybrid_search import handle_rrf_search, normalize, handle_weighted_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: normalize
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="List of scores to normalize"
    )

    # Command: weighted-search
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted hybrid search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Search query"
    )
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for the semantic search component"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results to return"
    )

    # Command: rrf-search
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform an RRF hybrid search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="Search query"
    )
    rrf_search_parser.add_argument(
        "-k", type=int, default=60, help="RRF parameter k"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Enhancement method to apply to the query"
    )
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method to use for reranking the results"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = handle_weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results):
                print(f"{i + 1}. {result['document']['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"   {result['document']['document'][:100]}...\n")
        case "rrf-search":
            results = handle_rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method)
            print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['document']['title']}")
                if "rerank_score" in result:
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                elif "rerank_rank" in result:
                    print(f"   Re-rank Rank: {result['rerank_rank']}")
                elif "cross_encoder_score" in result:
                    print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                print(f"   {result['document']['document'][:100]}...\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()