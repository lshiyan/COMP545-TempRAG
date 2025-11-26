import torch
import argparse
import json
import os
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from prompts.text_repr_extraction import TEXT_REPR_EXTRACTION_PROMPT
from dotenv import load_dotenv
from const import get_embedding_model, get_llama_tokenizer_and_model

load_dotenv()

def detect_gpu(request_gpu: bool) -> bool:
    return bool(request_gpu and torch.cuda.is_available())

def load_edges(path: str) -> List[Tuple[str, ...]]:
    """
    Loads a file where each line contains a tab separated TKG edge.

    Args:
        path (str): Path to the TSV file.

    Returns:
        List[Tuple[str, ...]]: A list of tuples, one per line.
    """
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = tuple(line.split("\t"))
            if len(parts) != 4:
                print("ERROR: Tuple should be length 4.")
            facts.append(parts)
    return facts

def load_sentences(path: str) -> List[str]:
    """
    Loads a file and returns a list where each element is a line from the file,
    stripped of leading/trailing whitespace.

    Args:
        path (str): Path to the file.

    Returns:
        List[str]: A list of lines.
    """
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def embed_sentences(
    model: SentenceTransformer,
    sentences: List[str],
    batch_size: int = 128,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Returns float32 np.array of shape (N, D) with L2-normalized rows.
    """
    device = "cuda" if use_gpu else "cpu"
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    ).astype("float32")
    return embeddings

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool) -> faiss.Index:
    """
    Builds an IndexFlatIP for cosine-like search (embeddings are normalized).
    If GPU available & requested, constructs on CPU then warms up GPU resources,
    but we always return a CPU index for persistence.
    """
    d = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    cpu_index.add(embeddings)

    if use_gpu:
        # Optional warm-up of GPU resources; index remains on CPU
        _ = faiss.StandardGpuResources()
    return cpu_index

def save_artifacts(
    outdir: str,
    edges: List[Tuple],
    index: faiss.Index,
    sentences: List[str],
    save_embeddings: bool,
    embeddings: np.ndarray = None
):
    os.makedirs(outdir, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(outdir, "index.faiss")
    faiss.write_index(index, index_path)

    # Save metadata (ID-aligned with embeddings order: 0..N-1)
    metadata = []
    for i, tuple in enumerate(edges):
        h, r, t, ts = tuple
        metadata.append(
            {
                "id": i,
                "head": h,
                "relation": r,
                "tail": t,
                "timestamp": ts,
                "text": sentences[i],
            }
        )

    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Optionally save the embedding matrix
    if save_embeddings and embeddings is not None:
        np.save(os.path.join(outdir, "embeddings.npy"), embeddings)

    return

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index over a Temporal Knowledge Graph TSV."
    )
    parser.add_argument(
        "--edges",
        required=True,
        help="Path to tkg edges file.",
    )
    
    parser.add_argument(
        "--sentences",
        required=True,
        help="Path to sentences file",
    )
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to write index and artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for embedding and/or generation.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        default=False,
        help="Also save embeddings.npy.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU if available for embeddings/FAISS.",
    )
    args = parser.parse_args()

    use_gpu = detect_gpu(args.use_gpu)
    if args.use_gpu and not use_gpu:
        print("[Info] --use-gpu was requested but CUDA not available. Falling back to CPU.")

    print("[1/5] Loading TKG TSV...")
    edges = load_edges(args.edges)
    sentences = load_sentences(args.sentences)
    
    if not sentences:
        raise RuntimeError("No sentences found.")

    print(
        f"[2/5] Loading embedding model ({os.getenv('EMBEDDING_MODEL') or 'all-MiniLM-L6-v2'})..."
    )
    embedding_model = get_embedding_model()
    
    print(
        f"[3/5] Starting embeddings"
    )
    embeddings = embed_sentences(
        embedding_model, sentences, args.batch_size, use_gpu
    )  # L2-normalized float32

    print(f"[4/5] Building FAISS index (IndexFlatIP) over dim={embeddings.shape[1]}...")
    cpu_index = build_faiss_index(embeddings, use_gpu)

    print(f"[5/5] Saving artifacts to {args.outdir} ...")
    save_artifacts(args.outdir, cpu_index, sentences, args.save_embeddings, embeddings)

    print("Done.")
    print(f" - Index:      {os.path.join(args.outdir, 'index.faiss')}")
    print(f" - Metadata:   {os.path.join(args.outdir, 'metadata.json')}")
    if args.save_embeddings:
        print(f" - Embeddings: {os.path.join(args.outdir, 'embeddings.npy')}")
    print(f"Index size: {cpu_index.ntotal}") 

if __name__ == "__main__":
    main()
