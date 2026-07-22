import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Image Embedding CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: verify_image_embedding
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify the embedding of an image",
    )
    verify_parser.add_argument(
        "image_path",
        help="Path to the image file to verify embedding",
    )

    # Command: image_search
    image_search_parser = subparsers.add_parser(
        "image_search",
        help="Search for similar images using an image",
    )
    image_search_parser.add_argument(
        "image_path",
        help="Path to the image file to search with",
    )

    args = parser.parse_args()
    image_path = args.image_path

    match args.command:
        case "verify_image_embedding":
            if verify_image_embedding(image_path):
                print("Image embedding verified successfully.")
            else:
                print("Failed to verify image embedding.")
        case "image_search":
            results = image_search_command(image_path)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['similarity_score']:.3f})")
                print(f"    {result['description'][:100]}...")
                print()

if __name__ == "__main__":
    main()