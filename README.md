# VRF-PoW Simulation

This repository contains the numerical simulators for the paper-scale PoW,
VRF-PoW, and Green-PoW experiments. The public interface intentionally focuses
on simulation and CSV generation; plotting and figure-composition code are not
part of the simulation package.

## Paper

Seungmin Kim, Haeung Choi, Minho Yoon, and Heung-No Lee,
"VRF-PoW: Proof of Work Consensus With Verifiable Random Function,"
*IEEE Transactions on Network Science and Engineering*, vol. 13,
pp. 6293-6309, 2026.

- DOI: [10.1109/TNSE.2026.3657405](https://doi.org/10.1109/TNSE.2026.3657405)
- Paper: [IEEE Transactions on Network Science and Engineering](https://ieeexplore.ieee.org/document/11362952/)

## Citation

If you use this software in academic work, please cite the paper above. GitHub
can generate APA and BibTeX entries from the repository's `CITATION.cff` file.

## Structure

- `vrf_pow_sim/network.py`: topology, propagation latency, and APSP helpers
- `vrf_pow_sim/hash_power.py`: hash-power and attacker distributions
- `vrf_pow_sim/protocols.py`: PoW, VRF-PoW, and Green-PoW simulators
- `generate_network.py`: seeded KNN topology and APSP generator
- `run_simulations.py`: command-line experiment runner

The all-pairs shortest-path (APSP) matrix used by the simulator is not included
in this repository. Supply a NumPy matrix with `--apsp`; the matrix must be
square, measured in seconds, and ordered consistently with the simulated nodes.

## Generate the network

Generate the default 10,000-node topology and propagation matrix:

```bash
python generate_network.py
```

The defaults place 10,000 nodes uniformly in a two-dimensional box, connect
each node to its 32 nearest neighbors, assign 20 ms to every edge, and use seed
42. The command writes `apsp_seconds_10000.npy` and a metadata JSON containing
the generation parameters. The APSP matrix is approximately 800 MB.

A small development topology can be generated with:

```bash
python generate_network.py --nodes 100 --neighbors 8 --output apsp_seconds_100.npy
```

## Environment

The published experiments used Python 3.11. Install the numerical dependencies:

```bash
python -m pip install -r requirements.txt
```

For an exact tested dependency set, use the lock files:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-dev-lock.txt  # includes pytest
```

## Simulation parameters

The command-line defaults define the following paper-scale configuration:

| Parameter | Default |
| --- | ---: |
| Nodes | 10,000 (inferred from the APSP shape) |
| Generated blocks | 1,000,000 |
| Target block time | 600 s |
| Difficulty adjustment interval | 2,016 blocks |
| Initial PoW/post-PoW difficulty | 6,000,000 |
| Hash-power concentration cases | top 1%, 2%, 5%, 10%, 20%, 50% |
| Target share of each top group | 50% |
| VRF pre-PoW factor `k` | 0.0001 |
| Progressive timeout window | 2,763.10 to 4,144.65 s |
| Green-PoW runner-ups | 1 |
| Green-PoW fixed timeout | 2,763.10 s |

The VRF configurations used by the runner are:

| Output label | Initial selection probability `p` | `k` |
| --- | ---: | ---: |
| `adv10` | 0.2 | 0.0001 |
| `adv20` | 0.4 | 0.0001 |
| `adv30` | 0.8 | 0.0001 |

## Run

A small smoke run:

```bash
python run_simulations.py --apsp /path/to/apsp.npy --blocks 100 --top-pct 1 --protocol all
```

The paper-scale configuration is the default:

```bash
python run_simulations.py --apsp /path/to/apsp.npy
```

Results are written to `simulation_output/` as event CSVs, node CSVs, the exact
hash-power arrays, and a `manifest.json` containing seeds and parameters. The
runner does not reuse CSVs implicitly; each invocation writes the requested
configuration to the selected output directory.

### Output files

| File pattern | Contents |
| --- | --- |
| `pow_topX_events.csv` | block time, difficulty, stale count, cumulative energy |
| `vrf_adv*_topX_events.csv` | PoW fields plus progressive-timeout indicators |
| `green_topX_events.csv` | phase-A/B times, difficulties, stale count, energy, timeout indicators |
| `*_nodes.csv` | final reward and energy totals for every node |
| `hash_power_topX.npy` | exact hash-power sample used by the scenario |
| `manifest.json` | APSP path, seeds, parameters, realized top share, and output mapping |

## Full-scale validation

The default 10,000-node generator and a short all-protocol simulation were
validated end-to-end on July 15, 2026. On the validation machine, APSP
generation took 44.8-50.7 seconds, reached approximately 0.885 GiB peak working
memory, and produced an 800,000,128-byte NumPy file (approximately 0.745 GiB).
The measured maximum propagation delay was 0.88 seconds. Runtime and memory
depend on the CPU, operating system, SciPy version, and storage device. A full
1,000,000-block run takes substantially longer than the smoke test.

## Test

Install the test dependency and run the automated suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## License

This software is available under the [MIT License](LICENSE).
