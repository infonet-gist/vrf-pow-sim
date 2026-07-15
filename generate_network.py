"""Generate a seeded KNN topology and its propagation-time APSP matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrf_pow_sim import (
    build_knn_adjacency,
    place_nodes_uniform,
    save_unweighted_apsp_numpy,
)


def parse_args() -> argparse.Namespace:
    """Parse topology, latency, batching, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--box-size", type=float, default=1_000.0)
    parser.add_argument("--latency-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("apsp_seconds_10000.npy"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing APSP file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reject invalid topology settings before creating a potentially large file.
    if args.nodes <= 1:
        raise ValueError("nodes must be greater than one")
    if not 1 <= args.neighbors < args.nodes:
        raise ValueError("neighbors must satisfy 1 <= neighbors < nodes")
    if args.box_size <= 0 or args.latency_ms <= 0:
        raise ValueError("box size and latency must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # The seed fixes node placement; KNN construction is deterministic for the
    # resulting coordinates. Edge latency is constant, so APSP stores hop count
    # multiplied by the per-edge delay.
    positions = place_nodes_uniform(args.nodes, args.box_size, args.seed)
    adjacency = build_knn_adjacency(positions, args.neighbors)
    save_unweighted_apsp_numpy(
        adjacency,
        args.output,
        edge_latency_seconds=args.latency_ms / 1_000.0,
        batch_size=args.batch_size,
    )

    # Keep the generation parameters next to the matrix so an ignored local
    # cache can still be identified and regenerated later.
    metadata = {
        "nodes": args.nodes,
        "neighbors": args.neighbors,
        "box_size": args.box_size,
        "edge_latency_ms": args.latency_ms,
        "seed": args.seed,
        "dtype": "float64",
        "shape": [args.nodes, args.nodes],
        "apsp_file": args.output.name,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved APSP matrix to {args.output}")
    print(f"saved generation parameters to {metadata_path}")


if __name__ == "__main__":
    main()
