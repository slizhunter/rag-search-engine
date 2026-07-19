import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"

def llm_spellcheck(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def llm_rewrite(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def llm_expand(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below to include additional relevant keywords.

    Consider:
    - Add synonyms and related concepts that might appear in movie descriptions.
    - Keep expansions relevant and focused.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    If you cannot expand the query, output the original unchanged.
    Output only the expanded query text, nothing else.

    User query: "{query}"
    """
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def llm_rerank_individual(query: str, doc: dict) -> str:
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def llm_rerank_batch(query: str, docs: list[dict]) -> list[int]:
    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {docs}

    Return the movie IDs in order of relevance, best match first.

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Your response MUST be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    MANDATORY: The response must contain exactly {len(docs)} movie IDs. If you end up with a result with a length other than {len(docs)}, try again until you get the correct length.

    Example output:
    [75, 12, 34, 2, 1]

    Ranking:"""
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], timeout=30)
    return json.loads(response.choices[0].message.content)