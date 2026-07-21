import argparse

from lib.evaluation import evaluate_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    results = evaluate_command(limit=limit)
    

    print(f"k={limit}\n")
    for query, res in results.items():
        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {res['precision']:.4f}")
        print(f"    - Recall@{limit}: {res['recall']:.4f}")
        print(f"    - F1 Score: {res['f1_score']:.4f}")
        print(f"    - Retrieved: {', '.join(res['retrieved'])}")
        print(f"    - Relevant: {', '.join(res['relevant'])}") 
        print()

if __name__ == "__main__":
    main()