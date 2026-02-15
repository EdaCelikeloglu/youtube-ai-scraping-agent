# YouTube AI Scraping Agent

This project implements a deterministic AI-powered pipeline developed.

Given a YouTube URL of a song, the system identifies the artist, discovers the artist’s first studio album, scrapes lyrics for each song from Genius, and produces structured analytical outputs.

---

## Overview

The pipeline performs the following steps:

1. Extracts the artist name from a YouTube URL using Apify YouTube Scraper.
2. Queries an LLM to determine the artist’s first studio album and its track list.
3. For each song in the album:
   - Finds the corresponding Genius lyrics page using Apify Google Search Scraper.
   - Scrapes the full HTML page using Apify Web Scraper.
   - Extracts clean raw lyrics text.
4. Computes text and token-based metrics.
5. Generates:
   - A structured JSON analysis output.
   - A final MD5 hash based on embeddings of token counts.

---

## Requirements

- Python 3.9+
- Apify account
- OpenRouter account

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

Main dependencies include:

- requests
- python-dotenv
- apify-client
- beautifulsoup4
- tiktoken
- sentence-transformers
- einops
- torch

---

## Environment Variables

Create a `.env` file in the project root directory:

APIFY_API_KEY=your_apify_api_key  
OPENROUTER_API_KEY=your_openrouter_api_key  
OPENROUTER_MODEL=openai/gpt-5.2  

During development, a lighter model can be used:

OPENROUTER_MODEL=openai/gpt-4o-mini  

Before submission, the model must be set to `openai/gpt-5.2`.

---

## Usage

Run the script from the terminal.

Note: JSON and hash outputs are produced in separate executions, as required by the assessment.


### JSON Mode
Prints the full analysis output:

```
python Eda_Celikeloglu_YouTube_AI.py "https://www.youtube.com/watch?v=VIDEO_ID" json
```

### Hash Mode
Prints only the final MD5 hash:

```
python Eda_Celikeloglu_YouTube_AI.py "https://www.youtube.com/watch?v=VIDEO_ID" hash
```
---

## Execution Proof

The following screenshots demonstrate successful execution of the pipeline.

- JSON output execution
- Hash output execution

Screenshots are available in the `screenshots/` directory.

---


## JSON Output Structure

```json
{
  "artist": "Artist Name",
  "album_name": "Album Name",
  "songs": [
    {
      "name": "Song Name",
      "lyrics_length_chars": 1482,
      "lyrics_length_words": 264,
      "lyrics_length_tokens": 395,
      "tokens_per_word": 1.5,
      "lyrics_hash": "md5_hash_here"
    }
  ],
  "total_tokens_all_songs": 5047,
  "avg_tokens_per_song": 630.88
}
```
---


## Lyrics Extraction Notes

- Lyrics are scraped exclusively from Genius using Apify Web Scraper.
- The full HTML document is captured.
- Only text contained in elements marked with `data-lyrics-container="true"` is used.
- Content before the word “Lyrics” (such as contributors or translations) is ignored.
- If lyrics cannot be retrieved, metrics default to zero values and execution continues.

---

## Token and Hash Computation

- Tokenization uses `cl100k_base` via `tiktoken`.
- For each song:
  - Character count
  - Word count
  - Token count
  - Tokens-per-word ratio
  - MD5 hash of raw lyrics
- Token counts are concatenated into a single comma-separated string.
- An embedding is generated using `nomic-ai/nomic-embed-text-v1.5`.
- The embedding vector is formatted to 10 decimal places per value.
- The final MD5 hash is computed from the formatted embedding string.

---

## Error Handling and Determinism

- All external requests use explicit timeouts.
- Errors are caught and logged without terminating the pipeline.
- LLM temperature is set to zero for deterministic behavior.
- Song ordering always matches original album release order.

---

## Project Structure

- Eda_Celikeloglu_YouTube_AI.py   Main pipeline
- requirements.txt                Dependency list
- README.md                       Documentation
- .env                            Environment variables (not committed)

---


## Disclaimer

This project was created solely for the AIMultiple technical assessment and is not intended for public distribution.

---

## Author

Eda Çelikeloglu
