"""Compare a known original song and sampled/derivative song with Sony SampleID.

This is a Lastrada proof of concept. It does not claim legal-grade boundaries.
It chunks both songs, embeds every chunk with Sony's released pretrained model,
and reports the strongest cross-song candidate matches for human review.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch

from sampleid import SampleID


SR = 16_000


@dataclass
class Chunk:
    start: float
    end: float
    audio: np.ndarray


def load_chunks(path: str, chunk_seconds: float, hop_seconds: float) -> list[Chunk]:
    y, _ = librosa.load(path, sr=SR, mono=True)
    chunk_n = int(chunk_seconds * SR)
    hop_n = int(hop_seconds * SR)

    chunks: list[Chunk] = []
    for start_n in range(0, max(1, len(y) - chunk_n + 1), hop_n):
        end_n = start_n + chunk_n
        audio = y[start_n:end_n]
        if len(audio) < chunk_n:
            audio = np.pad(audio, (0, chunk_n - len(audio)))
        chunks.append(
            Chunk(
                start=start_n / SR,
                end=min(end_n, len(y)) / SR,
                audio=audio.astype(np.float32),
            )
        )

    # Handle files shorter than one chunk.
    if not chunks:
        audio = np.pad(y, (0, max(0, chunk_n - len(y))))[:chunk_n].astype(np.float32)
        chunks.append(Chunk(start=0.0, end=len(y) / SR, audio=audio))
    return chunks


def embed(model: SampleID, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
    outputs = []
    for i in range(0, len(chunks), batch_size):
        batch = np.stack([c.audio for c in chunks[i : i + batch_size]])
        x = torch.from_numpy(batch)
        with torch.inference_mode():
            z = model(x, audio=True)
        z = z.squeeze(1)
        z = torch.nn.functional.normalize(z, dim=-1)
        outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("original", help="Known source/original audio file")
    ap.add_argument("derivative", help="Known sampled/derivative audio file")
    ap.add_argument("--chunk-seconds", type=float, default=5.0)
    ap.add_argument("--hop-seconds", type=float, default=2.5)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    if args.hop_seconds <= 0 or args.chunk_seconds <= 0:
        raise SystemExit("chunk and hop sizes must be positive")

    print("Loading Sony SampleID pretrained checkpoint...")
    model = SampleID.load_checkpoint()
    model.eval()

    print("Chunking audio...")
    original_chunks = load_chunks(args.original, args.chunk_seconds, args.hop_seconds)
    derivative_chunks = load_chunks(args.derivative, args.chunk_seconds, args.hop_seconds)

    print(
        f"Embedding {len(original_chunks)} original chunks and "
        f"{len(derivative_chunks)} derivative chunks..."
    )
    o = embed(model, original_chunks)
    d = embed(model, derivative_chunks)

    # Embeddings are unit-normalized, so dot product is cosine similarity.
    sim = d @ o.T
    flat_idx = np.argsort(sim.ravel())[::-1]

    rows = []
    seen = set()
    for idx in flat_idx:
        di, oi = np.unravel_index(idx, sim.shape)
        # Avoid emitting exact duplicate window pairs.
        key = (di, oi)
        if key in seen:
            continue
        seen.add(key)

        dc = derivative_chunks[di]
        oc = original_chunks[oi]
        rows.append(
            {
                "similarity": float(sim[di, oi]),
                "original_start": oc.start,
                "original_end": oc.end,
                "derivative_start": dc.start,
                "derivative_end": dc.end,
            }
        )
        if len(rows) >= args.top_k:
            break

    print("\nTop candidate matching windows")
    print("similarity | original (s) | derivative (s)")
    print("-" * 58)
    for r in rows:
        print(
            f"{r['similarity']:.4f}    | "
            f"{r['original_start']:.2f}-{r['original_end']:.2f} | "
            f"{r['derivative_start']:.2f}-{r['derivative_end']:.2f}"
        )

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} candidate matches to {out}")


if __name__ == "__main__":
    main()
