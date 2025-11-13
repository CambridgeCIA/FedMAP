#!/usr/bin/env python3

import os
import torch

from src.tasks import Interval, eICU, CPRD, Example


def load_state_dict(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state_dict (dict) at {path}, got {type(state)}")
    return state


def run_validation_for_client(
    cid: int,
    global_state_dict,
    device: torch.device,
    server_round: int = 0,
):
    config = {
        "local_epochs": 0,
        "server-round": server_round,
    }

    print("\n" + "=" * 70)
    print(f"Validating global model on Client {cid}")
    print("=" * 70)

    client = Example(cid=cid, config=config, device=device)
    client.set_models(global_model=global_state_dict, cnnet_modules=None)

    avg_loss, metrics = client.validate()

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def main():
    global_model_path = "src/checkpoints/global_model_example.pt"
    server_round = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading global model from: {global_model_path}")
    global_state_dict = load_state_dict(global_model_path, device)

    all_results = {}

    for cid in range(7, 10):
        metrics = run_validation_for_client(
            cid=cid,
            global_state_dict=global_state_dict,
            device=device,
            server_round=server_round,
        )
        all_results[cid] = metrics

    print("\n" + "#" * 70)
    print("Validation summary for clients 7–9")
    print("#" * 70)

    for cid, m in all_results.items():
        print(f"\nClient {cid}:")
        print(f"  Loss: {m.get('loss', 'N/A'):.4f}")
        print(f"  Accuracy: {m.get('accuracy', 'N/A'):.4f}")
        print(f"  Balanced Accuracy: {m.get('balanced_accuracy', 'N/A'):.4f}")
        print(f"  ROC AUC: {m.get('roc_auc', 'N/A'):.4f}")
        print(f"  Confusion Matrix (tn, fp, fn, tp): "
              f"({m.get('tn', 'N/A')}, {m.get('fp', 'N/A')}, "
              f"{m.get('fn', 'N/A')}, {m.get('tp', 'N/A')})")


if __name__ == "__main__":
    main()
