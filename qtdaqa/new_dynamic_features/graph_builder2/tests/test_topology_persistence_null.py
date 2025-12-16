from pathlib import Path
import json

import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_null_v1 import (
    PersistenceNullTopologyModule,
    FEATURE_DIM,
)


def test_config_template_contains_dim_and_params() -> None:
    tmpl = PersistenceNullTopologyModule.config_template()
    assert tmpl["module"] == PersistenceNullTopologyModule.module_id
    params = tmpl["params"]
    assert params["constant_value"] == 0.0
    assert tmpl.get("notes", {}).get("feature_dim") == FEATURE_DIM
    comments = tmpl.get("param_comments", {})
    assert "constant_value" in comments
    assert "element_filters" in comments
    json.dumps(tmpl)  # should be serialisable for --create-feature-config


def test_validate_params_accepts_defaults() -> None:
    params = dict(PersistenceNullTopologyModule._metadata.defaults)
    PersistenceNullTopologyModule.validate_params(params)  # should not raise


def test_generate_topology_outputs_constant_frame(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    interface_dir = tmp_path / "iface"
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    for d in (dataset_dir, interface_dir, work_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    pdb_path = dataset_dir / "model.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")

    interface_path = interface_dir / "model.interface.txt"
    interface_path.write_text(
        "\n".join(
            [
                "c<A>r<1>R<GLY> 0 0 0",
                "c<B>r<2>R<ALA> 1 1 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module = PersistenceNullTopologyModule(params=dict(PersistenceNullTopologyModule._metadata.defaults))
    result = module.generate_topology(
        pdb_paths=[pdb_path],
        dataset_dir=dataset_dir,
        interface_dir=interface_dir,
        work_dir=work_dir,
        log_dir=log_dir,
        sort_artifacts=True,
        round_decimals=3,
    )

    assert result["success"] == 1
    assert not result["failures"]
    topo_csv = work_dir / "topology" / "model.topology.csv"
    assert topo_csv.exists()

    df = pd.read_csv(topo_csv)
    assert df.shape == (2, 1 + FEATURE_DIM)
    assert list(df["ID"]) == ["c<A>r<1>R<GLY>", "c<B>r<2>R<ALA>"]
    numeric = df.drop(columns=["ID"]).to_numpy()
    assert (numeric == 0.0).all()
