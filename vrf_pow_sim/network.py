"""Network topology and propagation-delay utilities."""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree


def place_nodes_uniform(
    n_nodes: int,
    box_size: float = 1_000.0,
    seed: int | None = None,
) -> np.ndarray:
    """Place nodes uniformly in a square two-dimensional region."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if box_size <= 0:
        raise ValueError("box_size must be positive")
    return np.random.default_rng(seed).random((n_nodes, 2)) * box_size


def build_knn_graph(
    positions: np.ndarray,
    k: int = 32,
) -> tuple[list[list[int]], np.ndarray]:
    """Build a symmetrized k-nearest-neighbor graph.

    The returned distance matrix is dense. For the paper's 10,000-node
    experiment, prefer loading a precomputed APSP cache instead of rebuilding
    the topology during every run.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_nodes, 2)")
    n_nodes = len(positions)
    if not 1 <= k < n_nodes:
        raise ValueError("k must satisfy 1 <= k < n_nodes")

    # This compatibility helper materializes all Euclidean distances. Use
    # build_knn_adjacency for paper-scale generation to avoid the dense matrix.
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

    adjacency_sets = [set() for _ in range(n_nodes)]
    for node, nearest in enumerate(neighbors):
        for neighbor in nearest:
            neighbor = int(neighbor)
            adjacency_sets[node].add(neighbor)
            adjacency_sets[neighbor].add(node)
    return [sorted(items) for items in adjacency_sets], distances


def build_knn_adjacency(
    positions: np.ndarray,
    k: int = 32,
    workers: int = -1,
) -> list[list[int]]:
    """Build a symmetrized KNN graph without a dense distance matrix."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_nodes, 2)")
    n_nodes = len(positions)
    if not 1 <= k < n_nodes:
        raise ValueError("k must satisfy 1 <= k < n_nodes")

    # Query k+1 candidates because each node is its own nearest neighbor.
    tree = cKDTree(positions)
    _, candidates = tree.query(positions, k=k + 1, workers=workers)
    # Insert both directions to model bidirectional peer connections. KNN is
    # not symmetric by itself, so some nodes can end with more than k peers.
    adjacency_sets = [set() for _ in range(n_nodes)]
    for node, nearest in enumerate(np.atleast_2d(candidates)):
        neighbors = [int(item) for item in nearest if int(item) != node][:k]
        if len(neighbors) != k:
            raise ValueError("could not identify k distinct neighboring nodes")
        for neighbor in neighbors:
            adjacency_sets[node].add(neighbor)
            adjacency_sets[neighbor].add(node)
    return [sorted(items) for items in adjacency_sets]


def edge_latencies_seconds(
    adjacency: list[list[int]],
    latency_ms: float = 20.0,
) -> list[list[tuple[int, float]]]:
    """Attach the same propagation latency to every undirected edge."""
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    latency_seconds = latency_ms / 1_000.0
    weighted = [[] for _ in adjacency]
    seen: set[tuple[int, int]] = set()
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            edge = tuple(sorted((node, int(neighbor))))
            if edge in seen:
                continue
            seen.add(edge)
            left, right = edge
            weighted[left].append((right, latency_seconds))
            weighted[right].append((left, latency_seconds))
    return weighted


def dijkstra_seconds(
    weighted_adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute shortest propagation times from one source node."""
    n_nodes = len(weighted_adjacency)
    if not 0 <= source < n_nodes:
        raise ValueError("source is outside the graph")
    distances = np.full(n_nodes, np.inf)
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distances[node]:
            continue
        for neighbor, weight in weighted_adjacency[node]:
            candidate = current_distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def apsp_dijkstra(
    weighted_adjacency: list[list[tuple[int, float]]],
    verbose: bool = True,
) -> np.ndarray:
    """Compute a dense APSP matrix with repeated heap-based Dijkstra runs."""
    n_nodes = len(weighted_adjacency)
    distances = np.empty((n_nodes, n_nodes), dtype=float)
    for source in range(n_nodes):
        distances[source] = dijkstra_seconds(weighted_adjacency, source)
        if verbose and (source % 50 == 0 or source == n_nodes - 1):
            print(f"[APSP] computed {source + 1}/{n_nodes}")
    return distances


def save_apsp_numpy(distances: np.ndarray, path: str | Path) -> None:
    """Store a dense propagation-time matrix in NumPy's binary format."""
    np.save(Path(path), np.asarray(distances, dtype=float))


def load_apsp_numpy(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load an APSP matrix, memory-mapped by default for large experiments."""
    return np.load(Path(path), mmap_mode="r" if mmap else None)


def save_unweighted_apsp_numpy(
    adjacency: list[list[int]],
    path: str | Path,
    edge_latency_seconds: float = 0.02,
    batch_size: int = 64,
    verbose: bool = True,
) -> None:
    """Compute hop-based APSP in batches and write directly to a NumPy file."""
    n_nodes = len(adjacency)
    if n_nodes == 0:
        raise ValueError("adjacency must contain at least one node")
    if edge_latency_seconds <= 0:
        raise ValueError("edge_latency_seconds must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Convert the adjacency list into a symmetric sparse matrix. Values are
    # irrelevant because shortest_path is invoked in unweighted (hop) mode.
    rows: list[int] = []
    columns: list[int] = []
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            if not 0 <= neighbor < n_nodes or neighbor == node:
                raise ValueError("adjacency contains an invalid neighbor")
            rows.append(node)
            columns.append(int(neighbor))
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(n_nodes, n_nodes),
    )
    graph = graph.maximum(graph.T).tocsr()

    # Write batches directly into a valid .npy file instead of holding the
    # full 800 MB paper-scale matrix in an additional in-memory array.
    output = open_memmap(
        Path(path), mode="w+", dtype=np.float64, shape=(n_nodes, n_nodes)
    )
    for start in range(0, n_nodes, batch_size):
        stop = min(start + batch_size, n_nodes)
        hops = shortest_path(
            graph,
            directed=False,
            unweighted=True,
            indices=np.arange(start, stop),
        )
        if not np.all(np.isfinite(hops)):
            raise ValueError("KNN graph is disconnected; APSP contains infinity")
        output[start:stop] = hops * edge_latency_seconds
        if verbose:
            print(f"[APSP] computed {stop}/{n_nodes}")
    output.flush()
