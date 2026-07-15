import json
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def run_command(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )


def test_network_generator_and_simulation_cli(tmp_path):
    apsp = tmp_path / "apsp.npy"
    output = tmp_path / "simulation"
    run_command(
        "generate_network.py",
        "--nodes",
        "12",
        "--neighbors",
        "3",
        "--batch-size",
        "4",
        "--output",
        str(apsp),
    )
    run_command(
        "run_simulations.py",
        "--apsp",
        str(apsp),
        "--output",
        str(output),
        "--blocks",
        "4",
        "--adjustment-interval",
        "2",
        "--top-pct",
        "1",
        "--protocol",
        "all",
        "--seed",
        "42",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["nodes"] == 12
    assert manifest["blocks"] == 4
    assert set(manifest["scenarios"][0]["results"]) == {
        "pow",
        "vrf_adv10",
        "vrf_adv20",
        "vrf_adv30",
        "green",
    }
