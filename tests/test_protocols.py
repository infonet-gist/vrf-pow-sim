from unittest.mock import patch

import numpy as np

from vrf_pow_sim import (
    calculate_submission_times,
    simulate_greenpow_runnerup_apsp,
    simulate_pow_with_network_apsp,
)


def test_equation_14_boundaries_and_monotonicity():
    proofs = np.array([0.0, 0.2, 0.5, 1.0])
    submissions = calculate_submission_times(proofs, 0.2, 10.0, 20.0)

    assert submissions[0] == 0.0
    assert submissions[1] == 0.0
    assert submissions[-1] == 20.0
    assert np.all(np.diff(submissions) >= 0.0)


def test_pow_is_reproducible_with_a_fixed_seed():
    hash_power = np.ones(8)
    apsp = np.zeros((8, 8))
    arguments = dict(
        num_blocks=10,
        difficulty=80.0,
        reward=1.0,
        target_time=10.0,
        adjustment_interval=2,
        hash_power=hash_power,
        apsp_seconds=apsp,
        seed=123,
    )

    first_events, first_nodes = simulate_pow_with_network_apsp(**arguments)
    second_events, second_nodes = simulate_pow_with_network_apsp(**arguments)
    assert first_events.equals(second_events)
    assert first_nodes.equals(second_nodes)


def test_greenpow_counts_newly_admitted_fallback_energy():
    sampled_node_counts = []
    samples = iter(
        [
            np.array([1.0, 2.0, 3.0]),  # warm-up phase A
            np.array([1.0]),  # warm-up phase B
            np.array([1.0, 2.0, 3.0]),  # recorded phase A
            np.array([10.0]),  # original runner-up phase B
            np.array([3.0, 1.0]),  # newly admitted timeout miners
        ]
    )

    def fixed_mining_times(_rng, _difficulty, hash_power):
        sampled_node_counts.append(len(hash_power))
        return next(samples).copy()

    with patch("vrf_pow_sim.protocols._mining_times", side_effect=fixed_mining_times):
        events, nodes = simulate_greenpow_runnerup_apsp(
            num_rounds=1,
            initial_difficulty_a=1.0,
            initial_difficulty_b=1.0,
            reward_a=1.0,
            reward_b=1.0,
            target_time_a=1.0,
            target_time_b=1.0,
            adjustment_interval=1,
            hash_power=np.ones(3),
            apsp_seconds=np.zeros((3, 3)),
            runner_up_count=1,
            timeout_limit=5.0,
            seed=0,
        )

    # Nodes 0 and 2 each add one second of fallback work after the timeout.
    np.testing.assert_allclose(nodes["Node Energy"], [2.0, 6.0, 3.0])
    assert events.loc[0, "Total Energy So Far"] == 11.0
    assert bool(events.loc[0, "Timeout Fallback Won"])
    # The original runner-up is absent from the fallback draw.
    assert sampled_node_counts == [3, 1, 3, 1, 2]
