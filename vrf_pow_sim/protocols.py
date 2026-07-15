"""Discrete-event simulators for PoW, VRF-PoW, and Green-PoW."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Accept either a seed or an existing generator without changing callers."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _validate_common(
    num_events: int,
    adjustment_interval: int,
    hash_power: np.ndarray,
    apsp_seconds: np.ndarray,
    genesis_source: int,
) -> np.ndarray:
    """Validate inputs shared by all protocol simulators."""
    hash_power = np.asarray(hash_power, dtype=float)
    if num_events <= 0 or adjustment_interval <= 0:
        raise ValueError("event count and adjustment interval must be positive")
    if hash_power.ndim != 1 or len(hash_power) == 0 or np.any(hash_power < 0):
        raise ValueError("hash_power must be a non-empty, non-negative vector")
    if not np.any(hash_power > 0):
        raise ValueError("at least one node must have positive hash power")
    if apsp_seconds.shape != (len(hash_power), len(hash_power)):
        raise ValueError("apsp_seconds shape must match the number of nodes")
    if not 0 <= genesis_source < len(hash_power):
        raise ValueError("genesis_source is outside the node range")
    return hash_power


def _mining_times(
    rng: np.random.Generator,
    difficulty: float,
    hash_power: np.ndarray,
) -> np.ndarray:
    """Sample independent exponential PoW completion times for all nodes."""
    if difficulty <= 0:
        raise ValueError("difficulty must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        scales = np.where(hash_power > 0, difficulty / hash_power, np.inf)
    return rng.exponential(scales)


def _adjust_difficulty(
    times: list[float],
    target_time: float,
    difficulty: float,
    interval: int,
) -> float:
    """Scale difficulty so the latest interval approaches the target mean."""
    average = float(np.mean(times[-interval:]))
    return difficulty * target_time / average if average > 0 else difficulty


def calculate_submission_times(
    proof_values: np.ndarray,
    initial_threshold: float,
    timeout_start: float,
    timeout_end: float,
) -> np.ndarray:
    """Return the earliest post-PoW entry time from paper Equation (14)."""
    proof_values = np.asarray(proof_values, dtype=float)
    if not 0 < initial_threshold < 1:
        raise ValueError("initial_threshold must be strictly between 0 and 1")
    if timeout_start < 0 or timeout_end < timeout_start:
        raise ValueError("timeout window is invalid")
    if np.any((proof_values < 0) | (proof_values > 1)):
        raise ValueError("proof values must be in [0, 1]")

    # Proofs already below p0 may enter post-PoW immediately. Larger proofs
    # enter progressively between timeout_start and timeout_end.
    submission_times = np.zeros_like(proof_values)
    delayed = proof_values > initial_threshold
    submission_times[delayed] = (
        timeout_start
        + (timeout_end - timeout_start)
        * np.log(proof_values[delayed] / initial_threshold)
        / np.log(1.0 / initial_threshold)
    )
    return submission_times


def simulate_pow_with_network_apsp(
    num_blocks: int,
    difficulty: float,
    reward: float,
    target_time: float,
    adjustment_interval: int,
    hash_power: np.ndarray,
    apsp_seconds: np.ndarray,
    genesis_source: int = 0,
    seed: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate baseline PoW with propagation delays and stale blocks."""
    hash_power = _validate_common(
        num_blocks, adjustment_interval, hash_power, apsp_seconds, genesis_source
    )
    rng = _generator(seed)
    n_nodes = len(hash_power)

    # Calibrate the initial difficulty with an unrecorded adjustment interval.
    warmup_times = [
        float(np.min(_mining_times(rng, difficulty, hash_power)))
        for _ in range(adjustment_interval)
    ]
    difficulty = _adjust_difficulty(
        warmup_times, target_time, difficulty, adjustment_interval
    )

    block_times: list[float] = []
    difficulties: list[float] = []
    stale_counts: list[int] = []
    cumulative_energy: list[float] = []
    node_rewards = np.zeros(n_nodes)
    node_energy = np.zeros(n_nodes)
    previous_winner = genesis_source

    for block_index in range(num_blocks):
        # Nodes start after receiving the previous winning block. Mining and
        # propagation are expressed relative to that block's creation time.
        start_times = np.asarray(apsp_seconds[previous_winner], dtype=float)
        finish_times = start_times + _mining_times(rng, difficulty, hash_power)
        winner = int(np.argmin(finish_times))
        block_time = float(finish_times[winner])

        # A miner stops when it finds its own candidate or receives the winner.
        # Candidates found before the winner arrives are counted as stale.
        winner_arrivals = block_time + np.asarray(apsp_seconds[winner], dtype=float)
        stop_times = np.minimum(finish_times, winner_arrivals)

        node_energy += np.maximum(0.0, stop_times - start_times) * hash_power
        node_rewards[winner] += reward
        stale = (finish_times < winner_arrivals) & (np.arange(n_nodes) != winner)

        block_times.append(block_time)
        difficulties.append(difficulty)
        stale_counts.append(int(stale.sum()))
        cumulative_energy.append(float(node_energy.sum()))
        previous_winner = winner

        # Difficulty uses generation times from the latest complete interval.
        if (block_index + 1) % adjustment_interval == 0:
            difficulty = _adjust_difficulty(
                block_times, target_time, difficulty, adjustment_interval
            )

    blocks = pd.DataFrame(
        {
            "Block Time": block_times,
            "Difficulty": difficulties,
            "Stale Count": stale_counts,
            "Total Energy So Far": cumulative_energy,
        }
    )
    nodes = pd.DataFrame({"Node Rewards": node_rewards, "Node Energy": node_energy})
    return blocks, nodes


def simulate_vrf_pow_apsp(
    num_blocks: int,
    initial_threshold: float,
    difficulty_vrf_pre: float,
    difficulty_vrf_post: float,
    block_reward: float,
    target_block_time: float,
    t_start: float,
    t_end: float,
    adjustment_interval: int,
    k: float,
    node_hash_power: np.ndarray,
    is_attacker: np.ndarray,
    is_attacker2: np.ndarray,
    apsp_seconds: np.ndarray,
    genesis_source: int = 0,
    seed: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate pre-PoW, VRF selection, progressive entry, and post-PoW."""
    node_hash_power = _validate_common(
        num_blocks,
        adjustment_interval,
        node_hash_power,
        apsp_seconds,
        genesis_source,
    )
    if not 0 < k < 1:
        raise ValueError("k must be strictly between 0 and 1")
    n_nodes = len(node_hash_power)
    is_attacker = np.asarray(is_attacker, dtype=bool)
    is_attacker2 = np.asarray(is_attacker2, dtype=bool)
    if is_attacker.shape != (n_nodes,) or is_attacker2.shape != (n_nodes,):
        raise ValueError("attacker flags must match the number of nodes")
    rng = _generator(seed)

    # Warm up post-PoW difficulty before collecting reported blocks. Network
    # propagation and adversarial retries are excluded from this calibration.
    warmup_times: list[float] = []
    for _ in range(adjustment_interval):
        pre_times = _mining_times(rng, difficulty_vrf_pre, node_hash_power)
        proofs = rng.random(n_nodes)
        submissions = calculate_submission_times(proofs, initial_threshold, t_start, t_end)
        post_starts = np.maximum(pre_times, submissions)
        post_finishes = post_starts + _mining_times(
            rng, difficulty_vrf_post, node_hash_power
        )
        warmup_times.append(float(np.min(post_finishes)))

    difficulty_vrf_post = _adjust_difficulty(
        warmup_times,
        target_block_time,
        difficulty_vrf_post,
        adjustment_interval,
    )
    difficulty_vrf_pre = difficulty_vrf_post * k

    block_times: list[float] = []
    difficulties: list[float] = []
    stale_counts: list[int] = []
    cumulative_energy: list[float] = []
    timeout_flags: list[bool] = []
    timeout_won_flags: list[bool] = []
    node_rewards = np.zeros(n_nodes)
    node_energy = np.zeros(n_nodes)
    previous_winner = genesis_source

    for block_index in range(num_blocks):
        # Pre-PoW begins after the previous winner reaches each node.
        start_times = np.asarray(apsp_seconds[previous_winner], dtype=float)
        pre_durations = _mining_times(rng, difficulty_vrf_pre, node_hash_power)
        proofs = rng.random(n_nodes)

        # A flagged Sybil-grinding miner repeats pre-PoW until it obtains an
        # immediately eligible proof; honest nodes evaluate one proof.
        for attacker in np.flatnonzero(is_attacker):
            while proofs[attacker] > initial_threshold:
                pre_durations[attacker] += _mining_times(
                    rng,
                    difficulty_vrf_pre,
                    node_hash_power[attacker : attacker + 1],
                )[0]
                proofs[attacker] = rng.random()

        # Equation (14) determines the earliest allowed post-PoW entry. A node
        # must also have completed pre-PoW, hence the element-wise maximum.
        submissions = calculate_submission_times(
            proofs, initial_threshold, t_start, t_end
        )
        pre_finishes = start_times + pre_durations
        post_starts = np.maximum(pre_finishes, submissions)
        post_finishes = post_starts + _mining_times(
            rng, difficulty_vrf_post, node_hash_power
        )

        # The first completed post-PoW is the block winner. Other candidates
        # completed before the winner arrives are stale blocks.
        winner = int(np.argmin(post_finishes))
        block_time = float(post_finishes[winner])
        winner_arrivals = block_time + np.asarray(apsp_seconds[winner], dtype=float)
        stop_times = np.minimum(post_finishes, winner_arrivals)
        selected = proofs <= initial_threshold
        continuously_active = selected | is_attacker | is_attacker2

        # Immediately selected or adversarial nodes remain continuously active.
        # Other nodes perform pre-PoW, wait for their progressive entry time,
        # and consume post-PoW energy only after entry.
        energy = np.zeros(n_nodes)
        energy[continuously_active] = (
            np.maximum(
                0.0,
                stop_times[continuously_active] - start_times[continuously_active],
            )
            * node_hash_power[continuously_active]
        )

        inactive = ~continuously_active
        pre_stops = np.minimum(pre_finishes[inactive], winner_arrivals[inactive])
        pre_work = np.maximum(0.0, pre_stops - start_times[inactive])
        post_work = np.maximum(0.0, stop_times[inactive] - post_starts[inactive])
        energy[inactive] = (pre_work + post_work) * node_hash_power[inactive]

        node_energy += energy
        node_rewards[winner] += block_reward
        stale = (post_finishes < winner_arrivals) & (np.arange(n_nodes) != winner)
        timeout_triggered = block_time > t_start

        block_times.append(block_time)
        difficulties.append(difficulty_vrf_post)
        stale_counts.append(int(stale.sum()))
        cumulative_energy.append(float(node_energy.sum()))
        timeout_flags.append(timeout_triggered)
        timeout_won_flags.append(timeout_triggered and submissions[winner] > 0)
        previous_winner = winner

        # Keep the pre-/post-PoW difficulty ratio k fixed after each adjustment.
        if (block_index + 1) % adjustment_interval == 0:
            difficulty_vrf_post = _adjust_difficulty(
                block_times,
                target_block_time,
                difficulty_vrf_post,
                adjustment_interval,
            )
            difficulty_vrf_pre = difficulty_vrf_post * k

    blocks = pd.DataFrame(
        {
            "Block Time": block_times,
            "Difficulty": difficulties,
            "Stale Count": stale_counts,
            "Total Energy So Far": cumulative_energy,
            "Timeout Triggered": timeout_flags,
            "Timeout Won": timeout_won_flags,
        }
    )
    nodes = pd.DataFrame({"Node Rewards": node_rewards, "Node Energy": node_energy})
    return blocks, nodes


def simulate_greenpow_runnerup_apsp(
    num_rounds: int,
    initial_difficulty_a: float,
    initial_difficulty_b: float,
    reward_a: float,
    reward_b: float,
    target_time_a: float,
    target_time_b: float,
    adjustment_interval: int,
    hash_power: np.ndarray,
    apsp_seconds: np.ndarray,
    runner_up_count: int = 1,
    genesis_source: int = 0,
    timeout_limit: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate alternating Green-PoW phases with runner-up phase-B miners."""
    hash_power = _validate_common(
        num_rounds, adjustment_interval, hash_power, apsp_seconds, genesis_source
    )
    if not 1 <= runner_up_count < len(hash_power):
        raise ValueError("runner_up_count must be between 1 and n_nodes - 1")
    rng = _generator(seed)
    n_nodes = len(hash_power)
    difficulty_a = float(initial_difficulty_a)
    difficulty_b = float(initial_difficulty_b)

    # Calibrate phases A and B separately before collecting reported rounds.
    warmup_a: list[float] = []
    warmup_b: list[float] = []
    for _ in range(adjustment_interval):
        finishes_a = _mining_times(rng, difficulty_a, hash_power)
        order = np.argsort(finishes_a)
        winner_time_a = float(finishes_a[order[0]])
        runner_ups = order[1 : runner_up_count + 1]
        finishes_b = finishes_a[runner_ups] + _mining_times(
            rng, difficulty_b, hash_power[runner_ups]
        )
        warmup_a.append(winner_time_a)
        warmup_b.append(float(np.min(finishes_b)) - winner_time_a)

    difficulty_a = _adjust_difficulty(
        warmup_a, target_time_a, difficulty_a, adjustment_interval
    )
    difficulty_b = _adjust_difficulty(
        warmup_b, target_time_b, difficulty_b, adjustment_interval
    )

    a_winner_times: list[float] = []
    a_cut_times: list[float] = []
    b_generation_times: list[float] = []
    round_times: list[float] = []
    difficulty_a_history: list[float] = []
    difficulty_b_history: list[float] = []
    stale_counts: list[int] = []
    cumulative_energy: list[float] = []
    timeout_flags: list[bool] = []
    timeout_won_flags: list[bool] = []
    node_rewards = np.zeros(n_nodes)
    node_energy = np.zeros(n_nodes)
    previous_winner = genesis_source

    for round_index in range(num_rounds):
        # Phase A is open to all nodes. The earliest candidate wins A, while
        # the following candidates determine the phase-B runner-up set.
        starts_a = np.asarray(apsp_seconds[previous_winner], dtype=float)
        finishes_a = starts_a + _mining_times(rng, difficulty_a, hash_power)
        order = np.argsort(finishes_a)
        winner_a = int(order[0])
        winner_time_a = float(finishes_a[winner_a])
        runner_ups = order[1 : runner_up_count + 1]
        cutoff_node = int(order[runner_up_count])
        cutoff_time = float(finishes_a[cutoff_node])
        non_runner_up = np.ones(n_nodes, dtype=bool)
        non_runner_up[runner_ups] = False
        node_rewards[winner_a] += reward_a

        # Selected runner-ups begin phase B after completing their A candidate.
        finishes_b = finishes_a[runner_ups] + _mining_times(
            rng, difficulty_b, hash_power[runner_ups]
        )
        local_winner = int(np.argmin(finishes_b))
        winner_b = int(runner_ups[local_winner])
        winner_time_b = float(finishes_b[local_winner])
        timeout_triggered = timeout_limit is not None and winner_time_b > timeout_limit
        timeout_won = False

        # If no runner-up finishes B before the fixed timeout, nodes that were
        # not already mining phase B become eligible. Existing runner-ups keep
        # their original completion times instead of receiving a second clock.
        if timeout_triggered:
            fallback_finishes = np.full(n_nodes, np.inf)
            fallback_finishes[non_runner_up] = timeout_limit + _mining_times(
                rng,
                difficulty_b,
                hash_power[non_runner_up],
            )
            fallback_winner = int(np.argmin(fallback_finishes))
            if fallback_finishes[fallback_winner] < winner_time_b:
                winner_b = fallback_winner
                winner_time_b = float(fallback_finishes[fallback_winner])
                timeout_won = True

        node_rewards[winner_b] += reward_b
        winner_b_arrivals = winner_time_b + np.asarray(
            apsp_seconds[winner_b], dtype=float
        )
        cutoff_arrivals = cutoff_time + np.asarray(
            apsp_seconds[cutoff_node], dtype=float
        )

        # Non-runner-ups stop phase-A work after learning the cutoff candidate;
        # runner-ups stay active until the phase-B winner reaches them.
        stops_a = np.minimum(finishes_a, cutoff_arrivals)
        node_energy[non_runner_up] += (
            np.maximum(0.0, stops_a[non_runner_up] - starts_a[non_runner_up])
            * hash_power[non_runner_up]
        )
        runner_up_stops = winner_b_arrivals[runner_ups]
        node_energy[runner_ups] += (
            np.maximum(0.0, runner_up_stops - starts_a[runner_ups])
            * hash_power[runner_ups]
        )
        if timeout_triggered:
            # At the fixed timeout, every non-runner-up joins phase B. Each
            # newly admitted miner works until it finds a candidate itself or
            # receives the final phase-B winner, whichever happens first.
            fallback_stops = np.minimum(fallback_finishes, winner_b_arrivals)
            node_energy[non_runner_up] += (
                np.maximum(
                    0.0,
                    fallback_stops[non_runner_up] - timeout_limit,
                )
                * hash_power[non_runner_up]
            )

        # Phase-A candidates that appear before cutoff propagation completes are
        # stale unless they belong to the accepted winner/runner-up set.
        cutoff_arrivals = cutoff_time + np.asarray(
            apsp_seconds[cutoff_node], dtype=float
        )
        stale = finishes_a < cutoff_arrivals
        stale[order[: runner_up_count + 1]] = False
        winner_a_arrivals = winner_time_a + np.asarray(
            apsp_seconds[winner_a], dtype=float
        )
        stale[runner_ups] |= finishes_a[runner_ups] < winner_a_arrivals[runner_ups]
        stale[winner_a] = False

        a_winner_times.append(winner_time_a)
        a_cut_times.append(cutoff_time)
        b_generation_times.append(max(0.0, winner_time_b - winner_time_a))
        round_times.append(winner_time_b)
        difficulty_a_history.append(difficulty_a)
        difficulty_b_history.append(difficulty_b)
        stale_counts.append(int(stale.sum()))
        cumulative_energy.append(float(node_energy.sum()))
        timeout_flags.append(timeout_triggered)
        timeout_won_flags.append(timeout_won)
        previous_winner = winner_b

        # Adjust the two phase difficulties from their own recent durations.
        if (round_index + 1) % adjustment_interval == 0:
            difficulty_a = _adjust_difficulty(
                a_winner_times,
                target_time_a,
                difficulty_a,
                adjustment_interval,
            )
            difficulty_b = _adjust_difficulty(
                b_generation_times,
                target_time_b,
                difficulty_b,
                adjustment_interval,
            )

    rounds = pd.DataFrame(
        {
            "A_winner_time": a_winner_times,
            "A_cut_time": a_cut_times,
            "B_gen_time": b_generation_times,
            "Round_time": round_times,
            "diffA": difficulty_a_history,
            "diffB": difficulty_b_history,
            "Stale Count": stale_counts,
            "Total Energy So Far": cumulative_energy,
            "Timeout Triggered": timeout_flags,
            "Timeout Fallback Won": timeout_won_flags,
        }
    )
    nodes = pd.DataFrame({"Node Rewards": node_rewards, "Node Energy": node_energy})
    return rounds, nodes
