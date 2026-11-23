import argparse
import os
import yaml
import pandas as pd
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DeepGeneAIID


def run_prediction(config: dict, checkpoint_path: str, output_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(config["data_dir"], "dataset.pt")
    data = torch.load(data_path, map_location=device)

    model = DeepGeneAIID(
        gen_dim=data["x_gen"].size(1),
        exp_dim=data["x_exp"].size(1),
        stru_dim=data["x_stru"].size(1),
        hidden_dim=config.get("hidden_dim", 128),
        latent_dim=config.get("latent_dim", 64),
        dropout=config.get("dropout", 0.3),
        grl_lambda=config.get("grl_lambda", 1.0),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits, _ = model(
            data["x_gen"],
            data["x_exp"],
            data["x_stru"],
            data["ppi_edge_index"],
            data["dir_edge_index"],
            data.get("ppi_edge_weight"),
            data.get("dir_edge_weight"),
        )
        scores = torch.sigmoid(logits).cpu().numpy()

    gene_symbols = data.get("gene_symbols", [f"Gene_{i}" for i in range(len(scores))])
    df = pd.DataFrame({
        "Gene_symbol": gene_symbols,
        "Score": scores,
    }).sort_values(by="Score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} prioritized genes to {output_csv}")
    print(f"Genes with score >= 0.95: {(df['Score'] >= 0.95).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genome-wide AAG prioritization with DeepGene-AIID")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Trained model checkpoint")
    parser.add_argument("--output", type=str, default="results/prioritized_genes.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_prediction(cfg, args.checkpoint, args.output)
