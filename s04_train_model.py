import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from torch.amp import GradScaler, autocast

from gstat_architecture import HeteroGSTAT

BASE_DIR = r"E:\Genomic"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROC_DIR = os.path.join(BASE_DIR, "processed")
MODEL_PATH = os.path.join(OUTPUT_DIR, "explicit_hetero_gstat_best.pth")

BATCH_SIZE = 1024
NUM_EPOCHS = 20
INITIAL_LR = 0.001

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def run_hetero_trainer():
    print("Initiating diagnostic-grade heterogeneous GNN training...")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware optimization: {device.type.upper()}")

    features = torch.load(os.path.join(PROC_DIR, "hetero_features.pt"), weights_only=True)
    labels = torch.load(os.path.join(PROC_DIR, "hetero_labels.pt"), weights_only=True)
    edges = torch.load(os.path.join(PROC_DIR, "hetero_edge_indices.pt"), weights_only=True)

    data = HeteroData()
    data['variant'].x = features['variant']
    data['gene'].x = features['gene']
    data['variant'].y = labels['variant'].float().view(-1, 1)
    
    forward_edge = edges[('variant', 'affects', 'gene')]
    data['variant', 'affects', 'gene'].edge_index = forward_edge
    reverse_edge = torch.stack([forward_edge[1], forward_edge[0]], dim=0)
    data['gene', 'rev_affects', 'variant'].edge_index = reverse_edge

    actual_var_dim = data['variant'].x.shape[1]
    if actual_var_dim != 15:
        print(f"FATAL: Tensor dimension is {actual_var_dim}, expected 15.")
        sys.exit(1)

    y_raw = data['variant'].y.squeeze()
    valid_nodes = (y_raw == 0) | (y_raw == 1)
    valid_indices = torch.where(valid_nodes)[0]
    
    perm = torch.randperm(valid_indices.size(0))
    valid_indices = valid_indices[perm]
    
    split_idx = int(0.85 * len(valid_indices))
    train_idx = valid_indices[:split_idx]
    val_idx = valid_indices[split_idx:]
    
    torch.save(train_idx, os.path.join(PROC_DIR, "train_idx.pt"))
    torch.save(val_idx, os.path.join(PROC_DIR, "val_idx.pt"))
    print("Train and validation masks saved to disk.")
    
    num_patho = (y_raw[train_idx] == 1).sum().item()
    num_benign = (y_raw[train_idx] == 0).sum().item()
    pos_weight = torch.tensor([num_benign / (num_patho + 1e-9)]).to(device)
    
    train_loader = NeighborLoader(
        data, num_neighbors=[15, 10, 5], batch_size=BATCH_SIZE,
        input_nodes=('variant', train_idx), shuffle=True, num_workers=0
    )
    val_loader = NeighborLoader(
        data, num_neighbors=[15, 10, 5], batch_size=BATCH_SIZE,
        input_nodes=('variant', val_idx), shuffle=False, num_workers=0
    )

    model = HeteroGSTAT(variant_dim=15, gene_dim=2, hidden_dim=64, heads=4, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    scaler = GradScaler('cuda')
    
    best_missense_f1 = 0.0
    
    print("\nStarting neural network training...")
    print("-" * 100)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch = time.time()
        model.train()
        total_loss, total_examples = 0, 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                logits = model(batch.x_dict, batch.edge_index_dict)
                logits = logits[:batch['variant'].batch_size] 
                target = batch['variant'].y[:batch['variant'].batch_size]
                loss = criterion(logits, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * batch['variant'].batch_size
            total_examples += batch['variant'].batch_size

        avg_loss = total_loss / total_examples

        model.eval()
        all_preds, all_targets, all_probs, all_is_missense = [], [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with autocast('cuda'):
                    logits = model(batch.x_dict, batch.edge_index_dict)
                    logits = logits[:batch['variant'].batch_size]
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(batch['variant'].y[:batch['variant'].batch_size].cpu())
                
                is_missense = batch['variant'].x[:batch['variant'].batch_size, 0].cpu()
                all_is_missense.append(is_missense)
        
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_is_missense = torch.cat(all_is_missense).numpy() == 1.0
        
        v_f1 = f1_score(all_targets, all_preds, zero_division=0)
        v_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets.flatten())) > 1 else 0.0
        
        if all_is_missense.sum() > 0:
            miss_targets = all_targets[all_is_missense]
            miss_preds = all_preds[all_is_missense]
            v_miss_f1 = f1_score(miss_targets, miss_preds, zero_division=0)
        else:
            v_miss_f1 = 0.0

        scheduler.step(v_miss_f1) 
        
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | F1: {v_f1:.4f} | AUC: {v_auc:.4f} | Missense F1: {v_miss_f1:.4f} | Time: {time.time()-t_epoch:.1f}s")

        if v_miss_f1 > best_missense_f1:
            best_missense_f1 = v_miss_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print("New best model saved.")
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("="*100)
    print(f"Training completed. Best missense F1: {best_missense_f1:.4f}")

if __name__ == "__main__":
    run_hetero_trainer()