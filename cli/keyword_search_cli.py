#!/usr/bin/env python3

import argparse

from lib.search_utils import BM25_K1
from lib.frequency_search import inverse_document_frequency_command, term_frequency_command
from lib.keyword_search import bm25_idf_command, bm25_tf_command, build_command, search_command

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

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a specific term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

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
        case "tfidf":
            tf = term_frequency_command(args.doc_id, args.term)
            idf = inverse_document_frequency_command(args.term)
            tfidf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document {args.doc_id}: {bm25_tf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()