import os, mimetypes, base64

from .hybrid_search import HybridSearch
from .search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_MULTIPLIER
)

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"

def describe_image_command(image: str, query: str) -> str:
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        image_data = f.read()

    prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary

    DO NOT return a response containing anything about "User Safety". THIS IS AN IMPERATIVE INSTRUCTION.
    If the response contains any mention of "User Safety", it should be ignored and not included in the rewritten query.

    """

    data_url = f"data:{mime};base64,{base64.b64encode(image_data).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")