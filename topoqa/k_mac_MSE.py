import torch

def read_best(path):
    ckpt = torch.load(path, map_location="cpu")
    callbacks = ckpt.get("callbacks", {})
    for cb_key, cb_state in callbacks.items():
        if isinstance(cb_state, dict) and "best_model_score" in cb_state:
            return float(cb_state["best_model_score"])
    raise KeyError(f"best_model_score not found in {path}")

print("K trained model ckpt:", read_best("model/topo_best_val_loss=0.14407.ckpt.DETERM"))
print("TopoQA reference model ckpt:", read_best("model/topoqa.ckpt"))

