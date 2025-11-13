#!/usr/bin/env python3
"""
Fine-tune global model with ICNN-based FedMAP prior on selected clients.

- Uses a trained global model as initialization.
- Loads saved ICNN modules.
- Runs a few local epochs of FedMAP fine-tuning.
- Records validation metrics via Example._record_performance().
- Loops through client_id = 4, 5, 6.
"""

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


def run_finetune_for_client(
    cid: int,
    global_state_dict,
    icnn_state_dict,
    device: torch.device,
    local_epochs: int = 5,
    server_round: int = 0,
):
    """
    Run FedMAP fine-tuning with ICNN prior for a single client.

    Returns:
        best_state_dict: fine-tuned local model weights
        contribution: FedMAP contribution score
        metrics: validation metrics dict
    """
    # Configuration passed into Example
    config = {
        "local_epochs": local_epochs,   
        "server-round": server_round,  
    }

    print("\n" + "=" * 70)
    print(f"Starting fine-tuning for Client {cid}")
    print("=" * 70)

    client = Example(cid=cid, config=config, device=device)
    client.set_models(global_model=global_state_dict, cnnet_modules=icnn_state_dict)


    best_state_dict, contribution = client.train()
    avg_loss, metrics = client.validate()

    print(f"\nClient {cid} | Contribution: {contribution:.6f}")
    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return best_state_dict, contribution, metrics


def main():
    global_model_path = "src/checkpoints/global_model_example.pt"
    icnn_modules_path = "src/checkpoints/icnn_modules.pt"

    local_epochs = 5
    server_round = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print(f"Loading global model from: {global_model_path}")
    global_state_dict = load_state_dict(global_model_path, device)

    print(f"Loading ICNN modules from: {icnn_modules_path}")
    icnn_state_dict = load_state_dict(icnn_modules_path, device)

    all_results = {}

    for cid in range(4, 7):
        _, _, metrics = run_finetune_for_client(
            cid=cid,
            global_state_dict=global_state_dict,
            icnn_state_dict=icnn_state_dict,
            device=device,
            local_epochs=local_epochs,
            server_round=server_round,
        )

        all_results[cid] = {
            "metrics": metrics,
        }


    print("\n" + "#" * 70)
    print("Fine-tuning summary for clients 4–6")
    print("#" * 70)
    for cid, res in all_results.items():
        m = res["metrics"]
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
