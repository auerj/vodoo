"""Semantic search with OpenAI embeddings for knowledge articles."""

import os
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

from vodoo.client import OdooClient

MODEL = "knowledge.article"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small dimension


def _get_db_path() -> Path:
    """Get the path to the embeddings database."""
    vodoo_dir = Path.home() / ".vodoo"
    vodoo_dir.mkdir(parents=True, exist_ok=True)
    return vodoo_dir / "embeddings.db"


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            write_date TEXT,
            name TEXT,
            body_snippet TEXT,
            embedding BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)
    return conn


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert embedding list to binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Convert binary blob to embedding list."""
    count = len(blob) // 4  # float is 4 bytes
    return list(struct.unpack(f"{count}f", blob))


def _strip_html(html: str) -> str:
    """Strip HTML tags and decode entities from text."""
    if not html:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _create_embedding_text(name: str, body: str) -> str:
    """Create text for embedding from article name and body."""
    clean_body = _strip_html(body)
    # Truncate body to reasonable length for embedding
    max_body_len = 8000  # Leave room for name
    if len(clean_body) > max_body_len:
        clean_body = clean_body[:max_body_len] + "..."
    return f"{name}\n\n{clean_body}"


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package is required for semantic search. "
            "Install with: pip install openai"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for semantic search")

    client = OpenAI(api_key=api_key)

    # OpenAI API has a limit on batch size, process in chunks
    all_embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for item in response.data:
            all_embeddings.append(item.embedding)

    return all_embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy package is required for semantic search. " "Install with: pip install numpy"
        ) from e

    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def _fetch_articles_metadata(client: OdooClient) -> list[dict[str, Any]]:
    """Fetch article IDs and write_dates from Odoo."""
    return client.search_read(
        MODEL,
        domain=[],
        fields=["id", "write_date"],
        limit=10000,  # Reasonable limit
    )


def _fetch_articles_full(client: OdooClient, ids: list[int]) -> list[dict[str, Any]]:
    """Fetch full article data for given IDs."""
    if not ids:
        return []
    return client.read(MODEL, ids, ["id", "name", "body", "write_date"])


def sync_embeddings(client: OdooClient, console: Any = None) -> dict[str, int]:
    """Sync embeddings cache with Odoo articles.

    Returns stats about what was synced.
    """
    conn = _get_connection()

    # Get all articles from Odoo with their write_dates
    odoo_articles = _fetch_articles_metadata(client)
    odoo_by_id = {a["id"]: a["write_date"] for a in odoo_articles}
    odoo_ids = set(odoo_by_id.keys())

    # Get cached articles
    cursor = conn.execute("SELECT id, write_date FROM articles")
    cached = {row[0]: row[1] for row in cursor.fetchall()}
    cached_ids = set(cached.keys())

    # Determine what needs updating
    new_ids = odoo_ids - cached_ids
    deleted_ids = cached_ids - odoo_ids
    changed_ids = {aid for aid in (odoo_ids & cached_ids) if odoo_by_id[aid] != cached[aid]}

    ids_to_embed = list(new_ids | changed_ids)

    stats = {
        "new": len(new_ids),
        "changed": len(changed_ids),
        "deleted": len(deleted_ids),
        "unchanged": len(odoo_ids) - len(new_ids) - len(changed_ids),
        "embedded": 0,
    }

    # Delete removed articles from cache
    if deleted_ids:
        placeholders = ",".join("?" * len(deleted_ids))
        conn.execute(f"DELETE FROM articles WHERE id IN ({placeholders})", list(deleted_ids))
        conn.commit()

    # Embed new/changed articles in batches
    if ids_to_embed:
        if console:
            console.print(f"[cyan]Embedding {len(ids_to_embed)} articles...[/cyan]")

        batch_size = 50
        for i in range(0, len(ids_to_embed), batch_size):
            batch_ids = ids_to_embed[i : i + batch_size]
            articles = _fetch_articles_full(client, batch_ids)

            if not articles:
                continue

            # Prepare texts for embedding
            texts = []
            article_data = []
            for article in articles:
                text = _create_embedding_text(
                    article.get("name", ""),
                    article.get("body", "") or "",
                )
                texts.append(text)
                article_data.append(article)

            # Get embeddings
            embeddings = _get_embeddings(texts)

            # Store in cache
            for article, embedding in zip(article_data, embeddings):
                body_snippet = _strip_html(article.get("body", "") or "")[:500]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO articles (id, write_date, name, body_snippet, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        article["id"],
                        article["write_date"],
                        article.get("name", ""),
                        body_snippet,
                        _embedding_to_blob(embedding),
                    ),
                )
            conn.commit()
            stats["embedded"] += len(articles)

    conn.close()
    return stats


def semantic_search(
    client: OdooClient,
    query: str,
    limit: int = 20,
    console: Any = None,
) -> list[dict[str, Any]]:
    """Perform semantic search on knowledge articles.

    1. Sync cache with Odoo (lazy sync)
    2. Embed the query
    3. Find most similar articles
    4. Return top N results with similarity scores
    """
    # Sync cache
    stats = sync_embeddings(client, console=console)

    if console and (stats["new"] > 0 or stats["changed"] > 0 or stats["deleted"] > 0):
        console.print(
            f"[dim]Cache sync: {stats['new']} new, {stats['changed']} changed, "
            f"{stats['deleted']} deleted, {stats['unchanged']} unchanged[/dim]"
        )

    # Embed query
    query_embedding = _get_embeddings([query])[0]

    # Load all cached articles and compute similarities
    conn = _get_connection()
    cursor = conn.execute("SELECT id, name, body_snippet, embedding FROM articles")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    # Calculate similarities
    results = []
    for row in rows:
        article_id, name, body_snippet, embedding_blob = row
        article_embedding = _blob_to_embedding(embedding_blob)
        similarity = _cosine_similarity(query_embedding, article_embedding)
        results.append(
            {
                "id": article_id,
                "name": name,
                "body_snippet": body_snippet,
                "similarity": similarity,
            }
        )

    # Sort by similarity (descending) and limit
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


def clear_cache() -> None:
    """Clear the embeddings cache."""
    db_path = _get_db_path()
    if db_path.exists():
        db_path.unlink()
