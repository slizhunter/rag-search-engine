import argparse

from lib.describe_image import describe_image_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the image file to describe",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to rewrite based on the image",
    )

    args = parser.parse_args()
    image = args.image
    query = args.query

    describe_image_command(image, query)

if __name__ == "__main__":
    main()