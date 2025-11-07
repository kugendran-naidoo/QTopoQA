from qtdaqa.new_dynamic_features.model_training import train_cli


def test_primary_metric_prefers_selection_when_available():
    summary = {
        "selection_primary_metric": "selection_metric",
        "best_selection_metric": 0.123,
        "best_val_loss": 0.987,
    }
    metric, value = train_cli._resolve_primary_metric_value(summary)
    assert metric == "selection_metric"
    assert value == 0.123


def test_primary_metric_falls_back_to_val_loss_if_selection_missing():
    summary = {
        "selection_primary_metric": "selection_metric",
        "best_selection_metric": None,
        "best_val_loss": 0.456,
    }
    metric, value = train_cli._resolve_primary_metric_value(summary)
    assert metric == "val_loss"
    assert value == 0.456


def test_primary_metric_falls_back_to_selection_when_val_loss_absent():
    summary = {
        "selection_primary_metric": "val_loss",
        "best_val_loss": None,
        "best_selection_metric": 0.222,
    }
    metric, value = train_cli._resolve_primary_metric_value(summary)
    assert metric == "selection_metric"
    assert value == 0.222
