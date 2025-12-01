from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from qtdaqa.new_dynamic_features.graph_builder2.modules.node.dssp_topo_merge_passthrough_v1 import (
    DSSPTopologyMergePassthrough,
)


def test_passthrough_merges_all_topology_columns(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    interface_dir = work_dir / "interface"
    topo_dir = work_dir / "topology"
    interface_dir.mkdir(parents=True, exist_ok=True)
    topo_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = dataset_dir / "sample.pdb"
    pdb_path.write_text("", encoding="utf-8")
    interface_path = interface_dir / "sample.interface.txt"
    interface_path.write_text("c<A>r<1>R<ALA> 0 0 0\n", encoding="utf-8")
    topo_path = topo_dir / "sample.topology.csv"
    topo_df = pd.DataFrame(
        {
            "ID": ["c<A>r<1>R<ALA>"],
            "f0": [1.0],
            "lap_feature": [2.0],
        }
    )
    topo_df.to_csv(topo_path, index=False)

    # Monkeypatch DSSP to avoid external dependency
    def fake_run_dssp(self, pdb_file):
        return pd.DataFrame(
            {
                "ID": ["c<A>r<1>R<ALA>"],
                "rasa": [0.1],
                "phi": [0.2],
                "psi": [0.3],
                "SS8_0": [1],
                **{f"SS8_{i}": [0] for i in range(1, 8)},
                "AA_0": [1],
                **{f"AA_{i}": [0] for i in range(1, 21)},
            }
        )

    monkeypatch.setattr("qtdaqa.new_dynamic_features.graph_builder2.lib.node_features.node_fea.run_dssp", fake_run_dssp)

    module = DSSPTopologyMergePassthrough()
    structure_map = {"sample": pdb_path}
    result = module.generate_nodes(
        dataset_dir=dataset_dir,
        structure_map=structure_map,
        interface_dir=interface_dir,
        topology_dir=topo_dir,
        work_dir=work_dir,
        log_dir=log_dir,
        sort_artifacts=True,
    )
    assert result["success"] == 1
    output_csv = work_dir / "node_features" / "sample.csv"
    df_out = pd.read_csv(output_csv)
    assert "lap_feature" in df_out.columns
    # 32 DSSP/basic + 2 topo columns (f0, lap_feature)
    assert df_out.shape[1] == 1 + 32 + 2
