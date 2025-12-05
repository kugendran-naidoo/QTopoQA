import pandas as pd
import numpy as np

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_laplacian_hybrid_v1 import (
    PersistenceLaplacianHybridModule,
    compute_features_for_residues,
)


def test_metadata_and_defaults():
    module = PersistenceLaplacianHybridModule()
    module.params["jobs"] = 1  # force sequential to avoid process pool constraints in test env
    meta = module.metadata()
    assert meta.module_id == "topology/persistence_laplacian_hybrid/v1"
    assert module.default_alias == "140D topology + 32D Laplacian"
    assert module._feature_dim == 172
    tmpl = PersistenceLaplacianHybridModule.config_template()
    notes = tmpl.get("notes", {})
    assert notes.get("feature_dim_total_default") == 172
    assert notes.get("feature_dim_ph_default") == 140
    assert notes.get("feature_dim_lap_default") == 32


def test_validate_params_rejects_bad_graph_mode():
    params = {"lap_graph_mode": "invalid"}
    try:
        PersistenceLaplacianHybridModule.validate_params(params)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid lap_graph_mode")


def test_validate_params_rejects_bad_weight():
    params = {"lap_edge_weight": "bad"}
    try:
        PersistenceLaplacianHybridModule.validate_params(params)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid lap_edge_weight")


def test_validate_params_accepts_defaults():
    params = dict(PersistenceLaplacianHybridModule._metadata.defaults)
    # Should not raise
    PersistenceLaplacianHybridModule.validate_params(params)


def test_generate_topology_combines_ph_and_lap(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    interface_dir = work_dir / "interface"
    interface_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = dataset_dir / "sample.pdb"
    pdb_path.write_text("", encoding="utf-8")
    interface_path = interface_dir / "sample.interface.txt"
    interface_path.write_text(
        "\n".join(
            [
                "c<A>r<1>R<ALA> 0.0 0.0 0.0",
                "c<B>r<2>R<GLY> 1.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module = PersistenceLaplacianHybridModule()
    ph_cols = list(module._ph_columns)
    ids = [
        "c<A>r<1>R<ALA>",
        "c<B>r<2>R<GLY>",
    ]
    ph_stub = pd.DataFrame([[0.0] * len(ph_cols) for _ in ids], columns=ph_cols)
    ph_stub.insert(0, "ID", ids)

    def fake_compute_features_for_residues(pdb_path_arg, residues_arg, config_arg, logger=None):
        return ph_stub.copy()

    monkeypatch.setattr(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_laplacian_hybrid_v1.compute_features_for_residues",
        fake_compute_features_for_residues,
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
    assert len(result["failures"]) == 0
    output_file = work_dir / "topology" / "sample.topology.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    expected_cols = ["ID"] + module._ph_columns + module._lap_columns
    assert df.columns.tolist() == expected_cols
    assert df.shape[0] == 2
    assert df.shape[1] == len(expected_cols)
