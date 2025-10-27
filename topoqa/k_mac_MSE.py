import torch

def read_best(path):
    ckpt = torch.load(path, map_location="cpu")
    callbacks = ckpt.get("callbacks", {})
    for cb_key, cb_state in callbacks.items():
        if isinstance(cb_state, dict) and "best_model_score" in cb_state:
            return float(cb_state["best_model_score"])
    raise KeyError(f"best_model_score not found in {path}")

k_model = "model/k_best_model.chkpt"
topoqa_model = "model/topoqa.ckpt"

print("\nK trained model ckpt:", read_best(k_model))
print("\nTopoQA reference model ckpt:", read_best(topoqa_model))

