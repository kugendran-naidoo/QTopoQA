#!/usr/bin/env python3
"""
Run topology extraction for a single PDB + interface file with optional tracing.

Usage:
  QTOPO_TOPO_TRACE=1 QTOPO_TOPO_TRACE_FILTER=<token> \
    python -m qtdaqa.new_dynamic_features.graph_builder2.scripts.trace_topology_single \
      --pdb-path /path/to/model.pdb \
      --interface-file /path/to/model.interface.txt \
      --output-csv /tmp/topology.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from ..lib.new_topological_features import (
    ResidueDescriptor,
    TopologicalConfig,
    TopologyTracer,
    compute_features_for_residues,
)

LOGGER = logging.getLogger("trace_topology_single")


def _load_residues(interface_path: Path) -> List[ResidueDescriptor]:
    residues: List[ResidueDescriptor] = []
    with interface_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            token = stripped.split()[0]
            residues.append(ResidueDescriptor.from_string(token))
    return residues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-path", required=True, type=Path, help="Path to PDB file.")
    parser.add_argument("--interface-file", required=True, type=Path, help="Interface file (builder-style).")
    parser.add_argument("--output-csv", required=True, type=Path, help="Destination topology CSV.")
    parser.add_argument(
        "--neighbor-distance", type=float, default=8.0, help="Neighbor radius (matches persistence_basic default)."
    )
    parser.add_argument(
        "--filtration-cutoff", type=float, default=8.0, help="Filtration cutoff (matches persistence_basic default)."
    )
    parser.add_argument("--min-persistence", type=float, default=0.01, help="Minimum persistence to keep.")
    parser.add_argument(
        "--dedup-sort",
        action="store_true",
        help="Enable deduplication/sorting of input coordinates before topology.",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Emit per-residue progress logs (useful when tracing a handful of residues).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pdb_path = args.pdb_path.resolve()
    interface_path = args.interface_file.resolve()
    output_csv = args.output_csv.resolve()

    residues = _load_residues(interface_path)
    if not residues:
        LOGGER.error("Interface file %s contained no residue descriptors.", interface_path)
        return 2

    config = TopologicalConfig(
        neighbor_distance=float(args.neighbor_distance),
        filtration_cutoff=float(args.filtration_cutoff),
        min_persistence=float(args.min_persistence),
        element_filters=(
            ("C",),
            ("N",),
            ("O",),
            ("C", "N"),
            ("C", "O"),
            ("N", "O"),
            ("C", "N", "O"),
        ),
        dedup_sort=bool(args.dedup_sort),
        workers=1,
        log_progress=bool(args.log_progress),
    )

    tracer = TopologyTracer.from_env(pdb_path)
    LOGGER.info(
        "Tracing %d residues from %s (trace=%s dir=%s filters=%s)",
        len(residues),
        pdb_path.name,
        tracer.enabled,
        tracer.config.trace_dir if hasattr(tracer, "config") else None,
        tracer.config.filters if hasattr(tracer, "config") else None,
    )

    frame = compute_features_for_residues(
        pdb_path,
        residues,
        config,
        logger=LOGGER,
        tracer=tracer,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    LOGGER.info("Topology written to %s", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
