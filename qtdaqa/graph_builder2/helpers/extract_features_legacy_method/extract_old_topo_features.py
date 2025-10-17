#!/usr/bin/env python3
"""Extract topology features using the legacy topo_feature implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from Bio.PDB import PDBParser
import pandas as pd

import sys
sys.path.insert(0, "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa/src")
from topo_feature import topo_fea

ElementFilter = Union[List[str], str]
DEFAULT_ELEMENT_TOKENS: List[str] = ["C", "N", "O", "CN", "CO", "NO", "CNO"]
DEFAULT_ELEMENT_FILTERS: List[List[str]] = [
    ["C"],
    ["N"],
    ["O"],
    ["C", "N"],
    ["C", "O"],
    ["N", "O"],
    ["C", "N", "O"],
]


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text before exiting on errors."""

    def error(self, message: str) -> None:
        help_text = self.format_help()
        self._print_message(help_text, sys.stdout)
        self.exit(2, f"{self.prog}: error: {message}\n")


def normalise_element_filters(values: Optional[Iterable[str]]) -> List[ElementFilter]:
    """Convert CLI element tokens into the format expected by topo_fea."""
    if not values:
        return [group.copy() for group in DEFAULT_ELEMENT_FILTERS]

    cleaned: List[ElementFilter] = []
    for token in values:
        trimmed = token.strip()
        if not trimmed:
            continue
        if trimmed.lower() == "all":
            return ["all"]
        letters = [char.upper() for char in trimmed if char.isalpha()]
        if not letters:
            continue
        unique_letters = list(dict.fromkeys(letters))
        cleaned.append(unique_letters)

    return cleaned or ["all"]


def parse_args(argv: Iterable[str]) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = HelpOnErrorArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the Dockground_MAF2 PDBs",
    )
    parser.add_argument(
        "--neighbor-distance",
        type=float,
        default=6.0,
        help="Neighborhood radius in Å (default: %(default)s)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Persistence cutoff for H0 bars (default: %(default)s)",
    )
    parser.add_argument(
        "--min-persistence",
        type=float,
        default=0.01,
        help="Minimum persistence to keep bars (default: %(default)s)",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=list(DEFAULT_ELEMENT_TOKENS),
        help="Element filters (default: C, N, O, CN, CO, NO, CNO)",
    )
    parser.add_argument(
        "--residues-per-file",
        type=int,
        default=0,
        help="If >0, limit to this many residues per file (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write CSV outputs",
    )
    args = parser.parse_args(list(argv))

    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory not found or not a directory: {args.dataset_dir}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"Output path exists and is not a directory: {args.output_dir}")

    return args, parser


def collect_defaults(parser: argparse.ArgumentParser) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        default = action.default
        if default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = default
    return defaults


def select_residues(pdb_path: Path, limit: int) -> list[str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", str(pdb_path))
    descriptors: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                seq, icode = residue.id[1], residue.id[2] if residue.id[2].strip() else " "
                desc = f"c<{chain.id}>r<{seq}>"
                if icode.strip():
                    desc += f"i<{icode}>"
                desc += f"R<{residue.get_resname()}>"
                descriptors.append(desc)
                if limit > 0 and len(descriptors) >= limit:
                    return descriptors
    return descriptors


def main(argv: Iterable[str]) -> int:
    args, parser = parse_args(argv)
    defaults = collect_defaults(parser)
    element_filters = normalise_element_filters(args.elements)
    element_filters_display = [
        "all" if filt == "all" else "".join(filt) for filt in element_filters
    ]

    log_path = Path("topo_features_run.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        def log_print(*objects, **kwargs):
            print(*objects, **kwargs)
            print(*objects, **kwargs, file=log_file)

        usage_text = parser.format_help().rstrip("\n")
        for line in usage_text.splitlines():
            log_print(line)

        log_print("=== topo feature extraction run ===")
        log_print(f"log_path: {log_path}")
        log_print("Defaults:")
        for key in sorted(defaults):
            log_print(f"  {key}: {defaults[key]!r}")

        runtime_params = vars(args).copy()
        runtime_params["element_filters"] = element_filters_display
        log_print("Runtime parameters:")
        for key in sorted(runtime_params):
            log_print(f"  {key}: {runtime_params[key]!r}")

        log_print("Configuration:")
        log_print(f"  neighbor_distance: {args.neighbor_distance}")
        log_print(f"  cutoff: {args.cutoff}")
        log_print(f"  min_persistence: {args.min_persistence}")
        log_print(f"  element_filters: {element_filters_display}")
        log_print(f"  residues_per_file: {args.residues_per_file}")

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        pdb_files = sorted(args.dataset_dir.rglob("*.pdb"))

        for pdb_path in pdb_files:
            residues = select_residues(pdb_path, args.residues_per_file)
            if not residues:
                log_print(f"Skipping {pdb_path}: no residues")
                continue

            fea = topo_fea(
                str(pdb_path),
                neighbor_dis=args.neighbor_distance,
                e_set=element_filters,
                res_list=residues,
                Cut=args.cutoff,
            )
            df = fea.cal_fea()
            output_path = output_dir / f"{pdb_path.stem}.topology.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            log_print(f"Wrote {output_path}")

        return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
