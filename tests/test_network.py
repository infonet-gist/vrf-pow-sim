import numpy as np

from vrf_pow_sim import (
    build_knn_adjacency,
    load_apsp_numpy,
    place_nodes_uniform,
    save_unweighted_apsp_numpy,
)


def test_seeded_knn_and_apsp_generation(tmp_path):
    first = place_nodes_uniform(40, box_size=100.0, seed=7)
    second = place_nodes_uniform(40, box_size=100.0, seed=7)
    np.testing.assert_array_equal(first, second)

    adjacency = build_knn_adjacency(first, k=4, workers=1)
    assert all(len(neighbors) >= 4 for neighbors in adjacency)
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            assert node in adjacency[neighbor]

    output = tmp_path / "apsp.npy"
    save_unweighted_apsp_numpy(
        adjacency,
        output,
        edge_latency_seconds=0.02,
        batch_size=7,
        verbose=False,
    )
    apsp = np.asarray(load_apsp_numpy(output, mmap=False))
    assert apsp.shape == (40, 40)
    assert np.all(np.isfinite(apsp))
    np.testing.assert_allclose(np.diag(apsp), 0.0)
    np.testing.assert_allclose(apsp, apsp.T)
    np.testing.assert_allclose(apsp / 0.02, np.round(apsp / 0.02))
