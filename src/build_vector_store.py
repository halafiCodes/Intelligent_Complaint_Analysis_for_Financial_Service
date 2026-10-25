from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd




def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int = 12_000,
    group_col: str = "Product",
    random_state: int = 42,
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"Missing required column: {group_col}")

    product_distribution = df[group_col].value_counts(normalize=True)
    sampled_dfs: list[pd.DataFrame] = []

    for product, proportion in product_distribution.items():
        n_samples = max(1, int(round(proportion * sample_size)))
        product_subset = df[df[group_col] == product]
        if len(product_subset) == 0:
            continue

        sampled = product_subset.sample(
            n=min(n_samples, len(product_subset)),
            random_state=random_state,
        )
        sampled_dfs.append(sampled)

    if not sampled_dfs:
        return df.head(0).copy()

    return pd.concat(sampled_dfs, ignore_index=True)


def _word_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def create_documents(
    sampled_df: pd.DataFrame,
    text_col: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    if text_col not in sampled_df.columns:
        raise KeyError(f"Text column not found: {text_col}")

    documents: list[dict] = []
    for idx, row in sampled_df.iterrows():
        raw_text = row.get(text_col)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        chunks = _word_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_id, chunk in enumerate(chunks):
            documents.append(
                {
                    "chunk_text": chunk,
                    "complaint_id": int(idx),
                    "product": row.get("Product"),
                    "company": row.get("Company"),
                    "state": row.get("State"),
                    "chunk_id": int(chunk_id),
                }
            )

    return documents


def generate_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: faiss. "
            "Try: pip install faiss-cpu (or use conda if pip wheels aren't available on your platform)."
        ) from exc

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, faiss


def detect_text_column(df: pd.DataFrame) -> str:
    for candidate in ("cleaned_narrative", "Consumer complaint narrative", "complaint_text"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "No complaint text column found. Expected one of: "
        "cleaned_narrative, Consumer complaint narrative, complaint_text"
    )


def build_from_parquet(parquet_path: Path, out_dir: Path, batch_size: int = 10_000) -> None:
    try:
        import faiss  # pyright: ignore[reportMissingImports]
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Parquet builds require pyarrow and faiss-cpu.") from exc

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet input not found: {parquet_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    parquet_file = pq.ParquetFile(parquet_path)
    required_columns = {"document", "embedding", "metadata"}
    available_columns = set(parquet_file.schema_arrow.names)
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(f"Parquet input is missing columns: {sorted(missing_columns)}")

    index = None
    metadata: list[dict] = []
    total_rows = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["document", "embedding", "metadata"],
    ):
        rows = batch.to_pylist()
        embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            continue
        if index is None:
            index = faiss.IndexFlatL2(int(embeddings.shape[1]))
        elif int(embeddings.shape[1]) != int(index.d):
            raise ValueError("Parquet embeddings have inconsistent dimensions")
        index.add(embeddings)

        for row in rows:
            item_metadata = row["metadata"] or {}
            metadata.append(
                {
                    "chunk_text": str(row["document"] or ""),
                    "complaint_id": item_metadata.get("complaint_id"),
                    "product": item_metadata.get("product"),
                    "company": item_metadata.get("company"),
                    "state": item_metadata.get("state"),
                    "chunk_id": item_metadata.get("chunk_index"),
                }
            )
        total_rows += len(rows)
        print(f"Processed {total_rows:,} rows")

    if index is None or index.ntotal == 0:
        raise ValueError("Parquet input contains no embeddings")
    if len(metadata) != int(index.ntotal):
        raise ValueError("Generated metadata count does not match FAISS index count")

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss_index.bin"))
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved index to: {out_dir / 'faiss_index.bin'}")
    print(f"Saved metadata to: {out_dir / 'metadata.pkl'}")


def build_from_csv(data_path: Path, out_dir: Path) -> None:
    root = repo_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded: {data_path}")
    print("Dataset shape:", df.shape)

    text_col = detect_text_column(df)
    print(f"Using text column: {text_col}")

    sampled_df = stratified_sample(df)
    print("Sampled shape:", sampled_df.shape)

    documents = create_documents(sampled_df, text_col=text_col)
    print("Total chunks:", len(documents))

    if not documents:
        raise ValueError(
            "No text chunks were created. "
            "Check that the complaint text column is non-empty and contains strings."
        )

    texts = [doc["chunk_text"] for doc in documents]
    embeddings = generate_embeddings(texts)

    index, faiss = build_faiss_index(embeddings)
    faiss.write_index(index, str(out_dir / "faiss_index.bin"))

    metadata = documents
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved index to: {out_dir / 'faiss_index.bin'}")
    print(f"Saved metadata to: {out_dir / 'metadata.pkl'}")
    print("Vector store built successfully.")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, help="Use precomputed embeddings from a Parquet file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows loaded per Parquet batch",
    )
    args = parser.parse_args()

    if args.parquet:
        build_from_parquet(args.parquet, root / "vector_store", args.batch_size)
    else:
        build_from_csv(root / "data" / "processed" / "filtered_complaints.csv", root / "vector_store")


if __name__ == "__main__":
    main()
