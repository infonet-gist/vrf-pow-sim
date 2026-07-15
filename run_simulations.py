"""Run the paper-scale simulations without plotting or post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vrf_pow_sim import (
    assign_attackers,
    assign_top_share_lognormal,
    load_apsp_numpy,
    simulate_greenpow_runnerup_apsp,
    simulate_pow_with_network_apsp,
    simulate_vrf_pow_apsp,
)
from vrf_pow_sim.hash_power import realized_top_share


# Each entry is (output label, initial VRF selection probability, pre-PoW ratio).
VRF_SCENARIOS = (
    ("adv10", 0.2, 1e-4),
    ("adv20", 0.4, 1e-4),
    ("adv30", 0.8, 1e-4),
)


def parse_args() -> argparse.Namespace:
    """Parse experiment scale, input matrix, protocol, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apsp", type=Path, default=Path("apsp_seconds_10000.npy"))
    parser.add_argument("--output", type=Path, default=Path("simulation_output"))
    parser.add_argument("--blocks", type=int, default=1_000_000)
    parser.add_argument("--target-time", type=float, default=600.0)
    parser.add_argument("--adjustment-interval", type=int, default=2_016)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-pct", type=float, nargs="+", default=[1, 2, 5, 10, 20, 50]
    )
    parser.add_argument(
        "--protocol",
        choices=("all", "pow", "vrf", "green"),
        default="all",
    )
    return parser.parse_args()


def save_result(
    output: Path,
    prefix: str,
    events,
    nodes,
) -> dict[str, str]:
    """Write event- and node-level tables and return their manifest entries."""
    event_path = output / f"{prefix}_events.csv"
    node_path = output / f"{prefix}_nodes.csv"
    events.to_csv(event_path, index=False)
    nodes.to_csv(node_path, index=False)
    return {"events": event_path.name, "nodes": node_path.name}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Memory mapping keeps the 10,000 x 10,000 propagation matrix on disk and
    # loads only rows touched by the simulators.
    apsp = load_apsp_numpy(args.apsp)
    n_nodes = apsp.shape[0]
    if apsp.shape != (n_nodes, n_nodes):
        raise ValueError("APSP matrix must be square")

    # Normalized total hash power is n_nodes, so target_time * n_nodes gives
    # the initial network difficulty. Timeout endpoints follow the configured
    # multiples of the target block interval.
    initial_difficulty = args.target_time * n_nodes
    timeout_start = args.target_time * np.log(10) * 2
    timeout_end = args.target_time * np.log(10) * 3
    attackers = assign_attackers(n_nodes, 0)
    manifest: dict[str, object] = {
        "apsp": str(args.apsp),
        "nodes": n_nodes,
        "blocks": args.blocks,
        "target_time": args.target_time,
        "adjustment_interval": args.adjustment_interval,
        "base_seed": args.seed,
        "timeout_start": timeout_start,
        "timeout_end": timeout_end,
        "scenarios": [],
    }

    # Use an independent, deterministic random stream for each concentration
    # case and each protocol while sharing the same hash-power sample.
    for scenario_index, top_pct in enumerate(args.top_pct):
        scenario_seed = args.seed + scenario_index * 100
        hash_power, sigma = assign_top_share_lognormal(
            total_hash_power=float(n_nodes),
            n_nodes=n_nodes,
            top_pct=top_pct,
            top_share=0.5,
            seed=scenario_seed,
        )
        hash_path = args.output / f"hash_power_top{top_pct:g}.npy"
        np.save(hash_path, hash_power)
        scenario: dict[str, object] = {
            "top_pct": top_pct,
            "hash_power_file": hash_path.name,
            "hash_power_seed": scenario_seed,
            "lognormal_sigma": sigma,
            "realized_top_share": realized_top_share(hash_power, top_pct),
            "results": {},
        }
        prefix = f"top{top_pct:g}"

        # Baseline PoW.
        if args.protocol in ("all", "pow"):
            events, nodes = simulate_pow_with_network_apsp(
                num_blocks=args.blocks,
                difficulty=initial_difficulty,
                reward=1.0,
                target_time=args.target_time,
                adjustment_interval=args.adjustment_interval,
                hash_power=hash_power,
                apsp_seconds=apsp,
                seed=scenario_seed + 1,
            )
            scenario["results"]["pow"] = save_result(
                args.output, f"pow_{prefix}", events, nodes
            )

        # VRF-PoW configurations share the topology and hash-power sample.
        if args.protocol in ("all", "vrf"):
            for variant_index, (label, probability, k) in enumerate(VRF_SCENARIOS):
                events, nodes = simulate_vrf_pow_apsp(
                    num_blocks=args.blocks,
                    initial_threshold=probability,
                    difficulty_vrf_pre=initial_difficulty * k,
                    difficulty_vrf_post=initial_difficulty,
                    block_reward=1.0,
                    target_block_time=args.target_time,
                    t_start=timeout_start,
                    t_end=timeout_end,
                    adjustment_interval=args.adjustment_interval,
                    k=k,
                    node_hash_power=hash_power,
                    is_attacker=attackers,
                    is_attacker2=attackers,
                    apsp_seconds=apsp,
                    seed=scenario_seed + 10 + variant_index,
                )
                scenario["results"][f"vrf_{label}"] = save_result(
                    args.output, f"vrf_{label}_{prefix}", events, nodes
                )

        # One Green-PoW round creates phase-A and phase-B blocks, hence half as
        # many rounds as the requested total number of generated blocks.
        if args.protocol in ("all", "green"):
            events, nodes = simulate_greenpow_runnerup_apsp(
                num_rounds=args.blocks // 2,
                initial_difficulty_a=initial_difficulty,
                initial_difficulty_b=initial_difficulty / n_nodes,
                reward_a=1.0,
                reward_b=1.0,
                target_time_a=args.target_time,
                target_time_b=args.target_time,
                adjustment_interval=args.adjustment_interval,
                hash_power=hash_power,
                apsp_seconds=apsp,
                runner_up_count=1,
                timeout_limit=timeout_start,
                seed=scenario_seed + 20,
            )
            scenario["results"]["green"] = save_result(
                args.output, f"green_{prefix}", events, nodes
            )

        # Rewrite after each concentration case so completed work is described
        # even when a later long-running case is interrupted.
        manifest["scenarios"].append(scenario)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"completed top {top_pct:g}%")


if __name__ == "__main__":
    main()
