import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR, MODELS_DIR, ARTIFACT_DIR,
    NODE_EMBEDDING_DIM
)
from baselines.kairos.extract_embeddings import extract_embeddings
from baselines.kairos.supervised_head import LinearClassifier, predict_gpu

def perturb_temporal_data(data, labels, order="normal"):
    """
    Perturbs the validation TemporalData by reordering events PER SOURCE NODE.
    This mimics the THyN audit where each subject's sequence is reversed or shuffled.
    """
    if order == "normal":
        return data, labels
        
    print(f"  Applying {order.upper()} perturbation to TemporalData...")
    new_data = copy.deepcopy(data)
    new_labels = labels.copy()
    
    # We want to swap the contents of the events for each src node.
    src_array = data.src.numpy()
    unique_srcs = np.unique(src_array)
    
    # Create permutation array initialized to identity
    perm = np.arange(len(src_array))
    
    for src in unique_srcs:
        # Indices where this src appears in the global timeline
        idx = np.where(src_array == src)[0]
        if len(idx) <= 1:
            continue
            
        if order == "reverse":
            perm[idx] = idx[::-1]
        elif order == "shuffle":
            perm[idx] = np.random.permutation(idx)
            
    # Apply permutation to all data fields except timestamps 
    # (timestamps must remain monotonically increasing for TGN loader, 
    # but the actual event content/dst/msg is swapped to happen at this time)
    new_data.src = data.src[perm]
    new_data.dst = data.dst[perm]
    new_data.msg = data.msg[perm]
    
    # We keep data.t the same so the global timeline isn't broken, 
    # only the order of actions for each actor is changed.
    
    new_labels = labels[perm]
    
    return new_data, new_labels


def main():
    parser = argparse.ArgumentParser(description="Audit KAIROS Temporal Sensitivity")
    parser.add_argument("--dataset", default="theia", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--train-labels", default=None, help="Suffix of the trained model (e.g. 'l1')")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    graphs_dir = GRAPHS_DIR / args.dataset
    models_dir = MODELS_DIR / args.dataset
    emb_dir = ARTIFACT_DIR / "embeddings" / args.dataset
    
    # 1. Load trained Supervised Head
    suffix = f"_{args.train_labels}" if args.train_labels else ""
    head_path = emb_dir.parent.parent / "models" / args.dataset / f"supervised_model{suffix}.pt"
    
    if not head_path.exists():
        print(f"ERROR: Trained supervised head not found at {head_path}")
        print(f"Please run: python -m baselines.kairos.supervised_head --dataset {args.dataset}" + 
              (f" --train-labels {args.train_labels}" if args.train_labels else ""))
        return
        
    head_state = torch.load(head_path, map_location=device)
    mean = head_state["mean"]
    std = head_state["std"]
    
    # In_dim can be inferred from mean
    in_dim = len(mean)
    head_model = LinearClassifier(in_dim).to(device)
    head_model.load_state_dict(head_state["model"])
    head_model.eval()
    
    # 2. Load Validation Data
    val_data_path = graphs_dir / "val.TemporalData.pt"
    val_labels_path = graphs_dir / "val_labels.npy"
    
    orig_data = torch.load(val_data_path, map_location="cpu", weights_only=False)
    orig_labels = np.load(val_labels_path)
    
    model_config = torch.load(models_dir / "model_config.pt", weights_only=False)
    max_node_num = model_config["max_node_num"]
    
    orders = ["normal", "reverse", "shuffle"]
    results = {}
    
    print("\n--- Running KAIROS Temporal Sensitivity Probe ---")
    
    for order in orders:
        print(f"\nEvaluating order: {order.upper()}")
        
        # Perturb
        data, labels = perturb_temporal_data(orig_data, orig_labels, order=order)
        
        # Reload fresh GNN model so memory states are reset
        model_parts = torch.load(models_dir / "model.pt", map_location="cpu", weights_only=False)
        memory, gnn, link_pred, neighbor_loader = model_parts
        assoc = torch.empty(max_node_num, dtype=torch.long)
        
        # Extract embeddings
        print("  Extracting embeddings through TGN...")
        embeddings, losses = extract_embeddings(
            data, memory, gnn, link_pred, neighbor_loader, assoc,
            device, NODE_EMBEDDING_DIM
        )
        
        # Include reconstruction loss if it was used in the model (in_dim check)
        if in_dim > embeddings.shape[1]:
            loss_clipped = np.clip(losses, 0, np.percentile(losses, 99.9))
            embeddings = np.hstack([embeddings, loss_clipped.reshape(-1, 1)])
            
        # Standardize using training mean/std
        valid_mask = labels >= 0
        X_valid = (embeddings[valid_mask] - mean) / std
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = labels[valid_mask]
        
        # Predict
        probs = predict_gpu(head_model, X_valid, device)
        auprc = average_precision_score(y_valid, probs)
        
        results[order] = auprc
        print(f"  AUPRC: {auprc:.4f}")
        
    print("\n--- Summary ---")
    normal_auprc = results["normal"]
    if normal_auprc > 0:
        print(f"Reversed Drop:  {normal_auprc - results['reverse']:.4f} (Score: {results['reverse']/normal_auprc:.3f})")
        print(f"Shuffled Drop:  {normal_auprc - results['shuffle']:.4f} (Score: {results['shuffle']/normal_auprc:.3f})")

if __name__ == "__main__":
    main()
