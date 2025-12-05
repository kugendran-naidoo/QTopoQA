from pathlib import Path
import pandas as pd
import numpy as np

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.lightweight_mol_v1 import (
    TopologyLightweightMoLModule,
    FEATURE_DIM_DEFAULT,
    PH_DIM_DEFAULT,
    LAP_DIM_DEFAULT,
)


def test_config_template_has_dim_notes() -> None:
    tmpl = TopologyLightweightMoLModule.config_template()
    notes = tmpl.get("notes", {})
    assert notes.get("feature_dim_total_default") == FEATURE_DIM_DEFAULT
    assert notes.get("feature_dim_ph_default") == PH_DIM_DEFAULT
    assert notes.get("feature_dim_lap_default") == LAP_DIM_DEFAULT
    params = tmpl.get("params", {})
    assert params["neighbor_distance"] == 8.0
    assert tmpl["module"] == TopologyLightweightMoLModule.module_id


def test_validate_params_accepts_defaults() -> None:
    params = dict(TopologyLightweightMoLModule._metadata.defaults)
    TopologyLightweightMoLModule.validate_params(params)


def test_generate_topology_sorts_and_emits_expected_dim(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    interface_dir = work_dir / "interface"
    interface_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = dataset_dir / "sample.pdb"
    pdb_path.write_text("", encoding="utf-8")
    iface_path = interface_dir / "sample.interface.txt"
    iface_path.write_text(
        "\n".join(
            [
                "c<B>r<2>R<GLY> 1.0 0.0 0.0",
                "c<A>r<1>R<ALA> 0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module = TopologyLightweightMoLModule()
    ph_cols = list(module._ph_columns)
    ids = [
        "c<A>r<1>R<ALA>",
        "c<B>r<2>R<GLY>",
    ]
    ph_stub = pd.DataFrame([[0.0] * len(ph_cols) for _ in ids], columns=ph_cols)
    ph_stub.insert(0, "ID", ids)

    def fake_compute_features_for_residues(pdb_path_arg, residues_arg, config_arg, logger=None):
        return ph_stub.copy()

    def fake_build_unweighted_adjacency(node_coords, node_chains, cutoff):
        n = len(node_coords)
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = adj[j, i] = 1.0
        return adj

    def fake_compute_laplacian_moments(adj, moment_orders, config):
        # raw moments mu1-4 = 1..4, centered kappa2-4 = 2..4
        return [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0]

    monkeypatch.setattr(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.lightweight_mol_v1.compute_features_for_residues",
        fake_compute_features_for_residues,
    )
    monkeypatch.setattr(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.lightweight_mol_v1.build_unweighted_adjacency",
        fake_build_unweighted_adjacency,
    )
    monkeypatch.setattr(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.lightweight_mol_v1.compute_laplacian_moments",
        fake_compute_laplacian_moments,
    )

    result = module.generate_topology(
        pdb_paths=[pdb_path],
        dataset_dir=dataset_dir,
        interface_dir=interface_dir,
        work_dir=work_dir,
        log_dir=log_dir,
        sort_artifacts=True,
    )
    assert result["success"] == 1
    output_file = work_dir / "topology" / "sample.topology.csv"
    df = pd.read_csv(output_file)
    assert df.shape[1] == FEATURE_DIM_DEFAULT + 1
    # IDs should be sorted lexicographically
    assert df["ID"].tolist() == sorted(ids)
    # Lap columns should be present and include num_nodes + moments
    lap_cols = module._lap_columns
    assert all(col in df.columns for col in lap_cols)
