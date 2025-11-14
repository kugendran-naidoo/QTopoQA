from qtdaqa.new_dynamic_features.model_inference.inference_topoqa_cpu import (
    _uses_legacy_edge_encoder,
)


def test_detects_legacy_edge_encoder_keys():
    state_dict = {
        "edge_embed.0.weight": 0,
        "edge_embed.0.bias": 0,
    }
    assert _uses_legacy_edge_encoder(state_dict) is True


def test_detects_modern_edge_encoder_keys():
    state_dict = {
        "edge_embed.0.0.weight": 0,
        "edge_embed.0.0.bias": 0,
        "edge_embed.0.3.weight": 0,
        "edge_embed.0.3.bias": 0,
    }
    assert _uses_legacy_edge_encoder(state_dict) is False
