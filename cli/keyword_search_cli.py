#!/usr/bin/env python3

import argparse

from lib.frequency_search import inverse_document_frequency_command, term_frequency_command
from lib.keyword_search import build_command, search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a specific term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a specific term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built and saved to disk.")
        case "tf":
            tf = term_frequency_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")
        case "idf":
            idf = inverse_document_frequency_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()