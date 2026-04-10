"""Embedding pipeline: index video scenes with Gemini Embedding 2 for
semantic search, duplicate detection, and clustering.

Usage:
    python examples/embedding_search.py <path-to-video>
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from reverse_engine import reverse_engineer
from reverse_engine.clips import extract_scene_clips
from reverse_engine.embeddings_store import EmbeddingStore
from reverse_engine.gemini.embeddings import embed_query, embed_scenes

load_dotenv()


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "./video.mp4"
    output_dir = "./output"
    out = Path(output_dir)
    embeddings_path = out / "embeddings.npz"

    # ── Step 1: Run existing pipeline ────────────────────────────
    print("=" * 60)
    print("PHASE 1: Running video indexing pipeline")
    print("=" * 60)

    result = reverse_engineer(
        source,
        output_dir=output_dir,
        insights={"transcript", "visual", "text", "audio"},
        whisper_model="small",
        gemini_model="gemini-2.5-flash",
    )

    scene_tuples = [(s.start, s.end) for s in result.scenes]

    # ── Step 2: Extract per-scene clips ──────────────────────────
    print()
    print("=" * 60)
    print("PHASE 2: Extracting per-scene video clips and audio")
    print("=" * 60)

    clips_dir = out / "clips"
    print(f"[1] Cutting {len(scene_tuples)} scene clips (480p video + audio)...")
    clips = extract_scene_clips(
        Path(source),
        scene_tuples,
        clips_dir,
        video_scale=480,
    )
    print(f"    Extracted {len(clips)} clips to {clips_dir}")

    # ── Step 3: Embed scenes ─────────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 3: Embedding scenes with Gemini Embedding 2")
    print("=" * 60)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)
    dimensions = 768

    print(f"[2] Embedding {len(result.scenes)} scenes ({dimensions}d)...")
    embeddings = embed_scenes(
        client,
        result.scenes,
        clips,
        transcript=result.transcript,
        dimensions=dimensions,
    )
    print(f"    Embedded {len(embeddings)} scenes")

    # ── Step 4: Save to store ────────────────────────────────────
    store = EmbeddingStore.from_embeddings(embeddings)
    store.save(embeddings_path)
    print(f"[3] Saved embeddings to {embeddings_path}")

    # ── Step 5: Demo search ──────────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 4: Semantic scene search")
    print("=" * 60)

    queries = [
        "someone welding or working with metal",
        "car engine being repaired",
        "people talking and discussing",
    ]

    for query in queries:
        print(f'\nQuery: "{query}"')
        query_vec = embed_query(client, query, dimensions=dimensions)
        results = store.search(query_vec, top_k=3)

        for r in results:
            scene = result.scenes[r["scene_index"]]
            label = ""
            for lbl in result.labels:
                if lbl.scene_index == r["scene_index"]:
                    label = lbl.caption
                    break

            print(
                f"  Scene {r['scene_index']:3d} "
                f"[{scene.start:.1f}s - {scene.end:.1f}s] "
                f"score={r['score']:.3f} "
                f"| {label[:80]}"
            )

    # ── Step 6: Find duplicates ──────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 5: Near-duplicate detection")
    print("=" * 60)

    dupes = store.find_duplicates(threshold=0.92)
    if dupes:
        print(f"Found {len(dupes)} near-duplicate pairs:")
        for a, b, score in dupes[:10]:
            print(f"  Scene {a} <-> Scene {b}  similarity={score:.3f}")
    else:
        print("No near-duplicates found (threshold=0.92)")

    # ── Step 7: Cluster scenes ───────────────────────────────────
    n_clusters = min(5, len(embeddings))
    if n_clusters >= 2:
        print()
        print("=" * 60)
        print(f"PHASE 6: Scene clustering ({n_clusters} clusters)")
        print("=" * 60)

        clusters = store.cluster(n_clusters)
        for cluster_id, scene_indices in sorted(clusters.items()):
            print(f"\n  Cluster {cluster_id}: {len(scene_indices)} scenes")
            for si in scene_indices[:5]:
                label = ""
                for lbl in result.labels:
                    if lbl.scene_index == si:
                        label = lbl.caption
                        break
                print(f"    Scene {si:3d} | {label[:70]}")
            if len(scene_indices) > 5:
                print(f"    ... and {len(scene_indices) - 5} more")


if __name__ == "__main__":
    main()
