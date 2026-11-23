import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DeepGeneAIID


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: str, device: torch.device):
    """
    Expected preprocessed .pt dictionary in data_dir/dataset.pt:
      - x_gen: (11058, D_gen)
      - x_exp: (11058, D_exp)
      - x_stru: (11058, D_stru)
      - ppi_edge_index: (2, E_ppi)
      - ppi_edge_weight: (E_ppi,)
      - dir_edge_index: (2, E_dir)
      - dir_edge_weight: (E_dir,)
      - labeled_indices: (1660,) indices of 660 positive + 1000 putative negative genes
      - labels: (1660,) binary labels (1 for known AAG, 0 for negative)
    """
    data_path = os.path.join(data_dir, "dataset.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {data_path}. "
            "Please place your preprocessed multimodal tensors in data/dataset.pt."
        )
    data = torch.load(data_path, map_location=device)
    return data


def train_5fold_cv(config: dict):
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(config["data_dir"], device)

    labeled_idx = data["labeled_indices"].cpu().numpy()
    labels = data["labels"].cpu().numpy()

    skf = StratifiedKFold(n_splits=config.get("n_splits", 5), shuffle=True, random_state=config.get("seed", 42))
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    aurocs, auprs = [], []
    best_overall_auroc = 0.0

    for fold, (train_pos, val_pos) in enumerate(skf.split(labeled_idx, labels), start=1):
        train_nodes = torch.tensor(labeled_idx[train_pos], dtype=torch.long, device=device)
        val_nodes = torch.tensor(labeled_idx[val_pos], dtype=torch.long, device=device)
        y_train = torch.tensor(labels[train_pos], dtype=torch.float32, device=device)
        y_val = labels[val_pos]

        model = DeepGeneAIID(
            gen_dim=data["x_gen"].size(1),
            exp_dim=data["x_exp"].size(1),
            stru_dim=data["x_stru"].size(1),
            hidden_dim=config.get("hidden_dim", 128),
            latent_dim=config.get("latent_dim", 64),
            dropout=config.get("dropout", 0.3),
            grl_lambda=config.get("grl_lambda", 1.0),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        criterion = nn.BCEWithLogitsLoss()

        best_fold_auroc = 0.0
        best_fold_aupr = 0.0
        best_state = None

        for epoch in range(1, config.get("epochs", 200) + 1):
            model.train()
            optimizer.zero_grad()

            logits, info = model(
                data["x_gen"],
                data["x_exp"],
                data["x_stru"],
                data["ppi_edge_index"],
                data["dir_edge_index"],
                data.get("ppi_edge_weight"),
                data.get("dir_edge_weight"),
            )

            cls_loss = criterion(logits[train_nodes], y_train)
            loss = (
                cls_loss
                + config.get("lambda_adv", 0.1) * info["adv_loss"]
                + config.get("lambda_ortho", 0.05) * info["ortho_loss"]
            )
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits, _ = model(
                    data["x_gen"],
                    data["x_exp"],
                    data["x_stru"],
                    data["ppi_edge_index"],
                    data["dir_edge_index"],
                    data.get("ppi_edge_weight"),
                    data.get("dir_edge_weight"),
                )
                val_probs = torch.sigmoid(val_logits[val_nodes]).cpu().numpy()
                auroc = roc_auc_score(y_val, val_probs)
                aupr = average_precision_score(y_val, val_probs)

                if auroc > best_fold_auroc:
                    best_fold_auroc = auroc
                    best_fold_aupr = aupr
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Fold {fold} | Best AUROC: {best_fold_auroc:.4f} | AUPR: {best_fold_aupr:.4f}")
        aurocs.append(best_fold_auroc)
        auprs.append(best_fold_aupr)

        torch.save(best_state, os.path.join(config["checkpoint_dir"], f"model_fold{fold}.pt"))
        if best_fold_auroc > best_overall_auroc:
            best_overall_auroc = best_fold_auroc
            torch.save(best_state, os.path.join(config["checkpoint_dir"], "best_model.pt"))

    print("-" * 50)
    print(f"5-Fold CV Mean AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"5-Fold CV Mean AUPR : {np.mean(auprs):.4f} +/- {np.std(auprs):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation for DeepGene-AIID")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_5fold_cv(cfg)
