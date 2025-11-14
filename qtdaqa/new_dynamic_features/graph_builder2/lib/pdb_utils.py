"""Helpers for configuring Bio.PDB parser behaviour at runtime."""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

LOG = logging.getLogger("graph_builder")

_QUIET_DEFAULT = True
_CURRENT_QUIET = _QUIET_DEFAULT


def configure_pdb_parser(warnings_enabled: bool) -> None:
    """
    Configure how Bio.PDB warnings are surfaced and how parsers are instantiated.

    Args:
        warnings_enabled: When True, run parsers with QUIET=False and ensure
            PDBConstructionWarning messages reach the log. When False, run with
            QUIET=True and suppress those warnings.
    """
    global _CURRENT_QUIET
    _CURRENT_QUIET = not warnings_enabled
    if warnings_enabled:
        warnings.simplefilter("default", PDBConstructionWarning)
        LOG.info("PDB warnings enabled (Bio.PDB.PDBConstructionWarning will be logged).")
    else:
        warnings.simplefilter("ignore", PDBConstructionWarning)
        LOG.info("PDB warnings suppressed (use --pdb-warnings to enable).")


def create_pdb_parser() -> PDBParser:
    """Return a PDBParser configured according to the current warning mode."""
    return PDBParser(QUIET=_CURRENT_QUIET)
