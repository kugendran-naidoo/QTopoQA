from pathlib import Path
import json

import numpy as np
import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_permutation_null_v2 import (
    PersistencePermutationNullTopologyModule,
    FEATURE_DIM,
    _append_permutation_log,
    _permute_frame,
)


def test_config_template_contains_defaults() -> None:
    tmpl = PersistencePermutationNullTopologyModule.config_template()
    assert tmpl["module"] == PersistencePermutationNullTopologyModule.module_id
    params = tmpl["params"]
    assert params["shuffle_scope"] == "per_protein"
    assert params["shuffle_mode"] == "per_row"
    assert params["seed"] == 1337
    assert params["allow_seed_override"] is False
    assert tmpl.get("notes", {}).get("feature_dim") == FEATURE_DIM
    json.dumps(tmpl)  # config template should be serialisable


def test_validate_params_accepts_defaults() -> None:
    params = dict(PersistencePermutationNullTopologyModule._metadata.defaults)
    PersistencePermutationNullTopologyModule.validate_params(params)


def test_permute_frame_per_row_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["c<A>r<1>R<GLY>", "c<A>r<2>R<ALA>", "c<B>r<3>R<SER>", "c<B>r<4>R<THR>"],
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [10.0, 20.0, 30.0, 40.0],
        }
    )
    numeric_cols = ["f0", "f1"]

    rng = np.random.default_rng(1337)
    perm = rng.permutation(len(frame))
    expected = frame[numeric_cols].to_numpy()[perm, :]

    rng2 = np.random.default_rng(1337)
    permuted, _ = _permute_frame(frame, numeric_cols, "per_row", rng2)

    assert np.array_equal(permuted[numeric_cols].to_numpy(), expected)
    assert list(permuted["ID"]) == list(frame["ID"])
    assert sorted(permuted["f0"]) == sorted(frame["f0"])
    assert sorted(permuted["f1"]) == sorted(frame["f1"])


def test_append_permutation_log(tmp_path: Path) -> None:
    log_path = tmp_path / "topology.log"
    _append_permutation_log(
        log_path,
        shuffle_scope="per_protein",
        shuffle_mode="per_row",
        seed=1337,
        rows=4,
        cols=2,
        perm_hash="abc123",
        seed_override=False,
    )
    content = log_path.read_text(encoding="utf-8")
    assert "Permutation null details" in content
    assert "scope=per_protein" in content
    assert "mode=per_row" in content
    assert "seed=1337" in content
    assert "perm_hash=abc123" in content
