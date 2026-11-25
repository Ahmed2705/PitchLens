"""Prepare a versioned detection dataset; raw and output paths are explicit."""

import argparse
import json
from pathlib import Path

from pitchlens.detection_data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Directory containing both approved raw exports")
    parser.add_argument("--output", type=Path, required=True, help="New, separate processed dataset directory")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "detection_sources.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(prepare(args.raw, args.output, config, args.seed), indent=2))


if __name__ == "__main__":
    main()
