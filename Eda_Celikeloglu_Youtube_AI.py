import sys
import os
import json
import requests
import hashlib
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer
from apify_client import ApifyClient
from bs4 import BeautifulSoup
from tqdm import tqdm

import logging
logging.getLogger("apify_client").setLevel(logging.ERROR)

from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


load_dotenv()

APIFY_API_KEY = os.getenv("APIFY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not APIFY_API_KEY:
    raise RuntimeError("APIFY_API_KEY is missing")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing")


APIFY_YOUTUBE_SCRAPER = "https://api.apify.com/v2/acts/streamers~youtube-scraper/run-sync-get-dataset-items"
APIFY_GOOGLE_SCRAPER = "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items"
OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------
# 1. YOUTUBE -> ARTIST
# --------------------------------------------------

def get_artist_from_youtube(youtube_url: str) -> str:
    payload = {
        "startUrls": [{"url": youtube_url}],
        "maxResults": 1
    }
    params = {"token": APIFY_API_KEY}

    response = requests.post(APIFY_YOUTUBE_SCRAPER, json=payload, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()
    if not data:
        raise RuntimeError("Artist could not be extracted from YouTube")

    return data[0].get("channelName", "Unknown Artist")


# --------------------------------------------------
# 2. ARTIST -> FIRST ALBUM + SONGS (LLM)
# --------------------------------------------------

def get_first_album_and_songs(artist: str) -> dict:
    prompt = (
        f"Find the first studio album of the artist '{artist}'. "
        "Return the album name and the full song list in original release order. "
        "Respond with JSON only in the following format:\n"
        "{"
        "\"album_name\": \"...\", "
        "\"songs\": [\"Song 1\", \"Song 2\"]"
        "}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    response = requests.post(OPENROUTER_CHAT, headers=headers, json=body, timeout=90)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= start:
        raise RuntimeError("LLM did not return a valid JSON object")

    parsed = json.loads(content[start:end])

    if "album_name" not in parsed or "songs" not in parsed:
        raise RuntimeError("LLM JSON missing required keys")

    return parsed


# --------------------------------------------------
# 3. SONG -> GENIUS URL
# --------------------------------------------------

def get_genius_url(song_name: str, artist: str) -> str:
    query = f"{song_name} {artist} lyrics site:genius.com"

    payload = {
        "queries": query,
        "resultsPerPage": 10,
        "maxPagesPerQuery": 1
    }

    params = {"token": APIFY_API_KEY}

    response = requests.post(APIFY_GOOGLE_SCRAPER, json=payload, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()

    for item in data:
        for result in item.get("organicResults", []):
            link = result.get("url") or result.get("link") or ""
            if "genius.com" in link:
                return link

    return ""


# --------------------------------------------------
# 4. GENIUS -> HTML (WORKING VERSION)
# --------------------------------------------------

def scrape_lyrics_from_genius(genius_url: str) -> str:
    if not genius_url:
        return ""

    client = ApifyClient(APIFY_API_KEY)

    try:
        run = client.actor("apify/web-scraper").call(run_input={
            "startUrls": [{"url": genius_url}],
            "useBrowser": True,
            "proxyConfiguration": {"useApifyProxy": True},
            "pageFunction": """
            async function pageFunction(context) {
                const html = document.documentElement.outerHTML;
                return { html };
            }
            """,
            "maxConcurrency": 1
        })

        dataset_id = run["defaultDatasetId"]
        items = list(client.dataset(dataset_id).iterate_items())

        if not items:
            return ""

        return items[0].get("html", "") or ""

    except Exception as e:
        print("WARNING: Lyrics scraping failed:", str(e))
        return ""


# --------------------------------------------------
# 5. EXTRACT LYRICS AFTER 'Lyrics'
# --------------------------------------------------

def extract_raw_lyrics_from_html(html: str) -> str:
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.select('[data-lyrics-container="true"]')

    if not blocks:
        return ""

    raw_text = "\n".join(
        block.get_text(separator="\n").strip()
        for block in blocks
    )

    lines = raw_text.splitlines()

    for i, line in enumerate(lines):
        if "lyrics" in line.lower():
            return "\n".join(lines[i + 1:]).strip()

    return raw_text.strip()


# --------------------------------------------------
# 6. METRICS
# --------------------------------------------------

def compute_song_metrics(raw_lyrics: str) -> dict:
    lyrics = raw_lyrics or ""

    chars = len(lyrics)
    words = len(lyrics.split()) if lyrics else 0

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(lyrics)) if lyrics else 0

    tokens_per_word = tokens / float(words) if words > 0 else 0.0
    lyrics_hash = hashlib.md5(lyrics.encode("utf-8")).hexdigest()

    return {
        "lyrics_length_chars": chars,
        "lyrics_length_words": words,
        "lyrics_length_tokens": tokens,
        "tokens_per_word": tokens_per_word,
        "lyrics_hash": lyrics_hash
    }


# --------------------------------------------------
# 7. EMBEDDING + HASH
# --------------------------------------------------

def build_concatenated_token_counts(songs_array: list) -> str:
    return ",".join(str(s.get("lyrics_length_tokens", 0)) for s in songs_array)


def compute_embedding_md5(token_counts_str: str) -> str:
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
        trust_remote_code=True
    )

    vec = model.encode([token_counts_str], normalize_embeddings=False)[0]
    formatted = ",".join(f"{float(x):.10f}" for x in vec.tolist())

    return hashlib.md5(formatted.encode("utf-8")).hexdigest()


# --------------------------------------------------
# 8. PIPELINE
# --------------------------------------------------

def run_analysis(youtube_url: str) -> dict:
    artist = get_artist_from_youtube(youtube_url)
    album_data = get_first_album_and_songs(artist)

    songs_output = []

    for song_name in tqdm(album_data["songs"], desc="Processing songs"):
        genius_url = get_genius_url(song_name, artist)

        with suppress_stdout_stderr():
            html = scrape_lyrics_from_genius(genius_url)

        lyrics = extract_raw_lyrics_from_html(html)
        m = compute_song_metrics(lyrics)

        songs_output.append({
            "name": song_name,
            **m
        })

    total_tokens = sum(s["lyrics_length_tokens"] for s in songs_output)
    avg_tokens = total_tokens / float(len(songs_output)) if songs_output else 0.0

    return {
        "artist": artist,
        "album_name": album_data["album_name"],
        "songs": songs_output,
        "total_tokens_all_songs": total_tokens,
        "avg_tokens_per_song": avg_tokens
    }


# --------------------------------------------------
# 9. CLI
# --------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 Eda_Celikeloglu_YouTube_AI.py <youtube_url> <json|hash>")
        sys.exit(1)

    youtube_url = sys.argv[1]
    mode = sys.argv[2].lower()

    try:
        analysis = run_analysis(youtube_url)

        if mode == "json":
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
            return

        if mode == "hash":
            token_str = build_concatenated_token_counts(analysis["songs"])
            print(compute_embedding_md5(token_str))
            return

        print("ERROR: mode must be 'json' or 'hash'")
        sys.exit(1)

    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
