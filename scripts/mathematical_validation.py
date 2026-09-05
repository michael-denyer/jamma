#!/usr/bin/env python3
"""Generate GEMMA references or compare a validation case family."""

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.math_validation import fixtures, loco_cases, pipeline_cases, weight_contract
from tests.math_validation.phase1 import compare_phase1
from tests.math_validation.supplied_cases import compare

FAMILIES = {
    "fixtures": (
        partial(fixtures.generate_reference, fixtures.load_manifest()),
        partial(compare, fixtures.load_manifest(), fixtures.REFERENCE),
    ),
    "pipeline": (pipeline_cases.generate_external, pipeline_cases.compare_pipeline),
    "loco": (loco_cases.generate_external, loco_cases.compare_loco),
    "weights": (weight_contract.generate_external, weight_contract.compare_weights),
    "phase1": (None, compare_phase1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Generate external GEMMA references"
    )
    generate.add_argument(
        "family", choices=[key for key, value in FAMILIES.items() if value[0]]
    )
    generate.add_argument("--gemma", type=Path, required=True)
    compare_parser = commands.add_parser("compare", help="Compare committed references")
    compare_parser.add_argument("family", choices=FAMILIES)
    for command in (generate, compare_parser):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator, comparator = FAMILIES[args.family]
    if args.command == "generate":
        assert generator is not None
        generator(args.output, args.gemma)
        return 0
    result = comparator(args.output)
    print(result["status"])
    return int(result["status"] != "VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
