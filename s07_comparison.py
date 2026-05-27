import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborLoader

try:
    from gstat_architecture import HeteroGSTAT
except ImportError:
    print("CRITICAL ERROR: Could not import HeteroGSTAT. Ensure gstat_architecture.py is in the same folder.")
    sys.exit(1)

BASE_DIR = r"E:\Genomic"
PROC_DIR = os.path.join(BASE_DIR, "processed")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


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


def load_golden_tensors():
    print("Loading tensors from compiler output...")
    try:
        features = torch.load(os.path.join(PROC_DIR, "hetero_features.pt"), weights_only=True)
        labels = torch.load(os.path.join(PROC_DIR, "hetero_labels.pt"), weights_only=True)
        edges = torch.load(os.path.join(PROC_DIR, "hetero_edge_indices.pt"), weights_only=True)
    except FileNotFoundError:
        print(f"ERROR: Tensors not found in {PROC_DIR}. Run Phase 2 compiler first.")
        sys.exit(1)

    variant_features = features['variant']
    gene_features = features['gene']
    y_raw = labels['variant']
    edge_index_v2g = edges[('variant', 'affects', 'gene')]

    valid_nodes = (y_raw == 0) | (y_raw == 1)
    valid_indices = torch.where(valid_nodes)[0]
    
    num_variants = variant_features.size(0)
    num_genes = gene_features.size(0)

    hetero_data = HeteroData()
    hetero_data['variant'].x = variant_features
    hetero_data['gene'].x = gene_features
    hetero_data['variant'].y = y_raw.float()
    hetero_data['variant', 'affects', 'gene'].edge_index = edge_index_v2g
    
    hetero_data = T.ToUndirected()(hetero_data)
    
    perm = torch.randperm(valid_indices.size(0))
    valid_indices = valid_indices[perm]
    split_idx = int(0.80 * len(valid_indices))
    train_idx = valid_indices[:split_idx]
    test_idx = valid_indices[split_idx:]
    
    for mask_name, idx in zip(['train_mask', 'test_mask'], [train_idx, test_idx]):
        mask = torch.zeros(num_variants, dtype=torch.bool)
        mask[idx] = True
        hetero_data['variant'][mask_name] = mask

    print("Executing data downgrade for homogeneous baselines...")
    padded_gene_features = torch.cat([gene_features, torch.zeros(num_genes, 13)], dim=1)
    homo_x = torch.cat([variant_features, padded_gene_features], dim=0)
    
    shifted_edge_index = edge_index_v2g.clone()
    shifted_edge_index[1, :] += num_variants
    homo_edge_index = torch.cat([shifted_edge_index, shifted_edge_index.flip(0)], dim=1)
    
    homo_data = Data(x=homo_x, edge_index=homo_edge_index)
    
    for mask_name in ['train_mask', 'test_mask']:
        homo_mask = torch.zeros(num_variants + num_genes, dtype=torch.bool)
        homo_mask[:num_variants] = hetero_data['variant'][mask_name]
        setattr(homo_data, mask_name, homo_mask)
        
    homo_data.y = torch.zeros(num_variants + num_genes, dtype=torch.float)
    homo_data.y[:num_variants] = y_raw.float()

    return hetero_data, homo_data, train_idx, test_idx


def run_xgboost(hetero_data, train_idx, test_idx):
    print("\n--- Running Baseline 1: XGBoost (Non-Graph) ---")
    X_train = hetero_data['variant'].x[train_idx].numpy()
    y_train = hetero_data['variant'].y[train_idx].numpy()
    X_test = hetero_data['variant'].x[test_idx].numpy()
    y_test = hetero_data['variant'].y[test_idx].numpy()
    
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, tree_method='hist', 
                            scale_pos_weight=pos_weight)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print_metrics("XGBoost", y_test, preds, probs)


class HomogeneousGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type="GCN"):
        super().__init__()
        if model_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == "GraphSAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x)


def print_metrics(model_name, y_true, y_pred, y_prob):
    print(f"Results for {model_name}:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"AUROC:     {roc_auc_score(y_true, y_prob):.4f}\n")


def train_and_eval_gnn(model, data, loader, is_hetero=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    if is_hetero:
        train_y = data['variant'].y[data['variant'].train_mask]
    else:
        train_y = data.y[data.train_mask]
        
    num_patho = (train_y == 1).sum().item()
    num_benign = (train_y == 0).sum().item()
    pos_weight = torch.tensor([num_benign / max(1, num_patho)]).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    
    model.train()
    for epoch in range(25):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if is_hetero:
                logits = model(batch.x_dict, batch.edge_index_dict)
                logits = logits[:batch['variant'].batch_size].squeeze()
                target = batch['variant'].y[:batch['variant'].batch_size]
                mask = batch['variant'].train_mask[:batch['variant'].batch_size]
            else:
                logits = model(batch.x, batch.edge_index).squeeze()
                target = batch.y
                mask = batch.train_mask
                
            loss = criterion(logits[mask], target[mask])
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    
    eval_loader = NeighborLoader(
        data, 
        num_neighbors=[-1], 
        batch_size=2048, 
        input_nodes=('variant', data['variant'].test_mask) if is_hetero else data.test_mask, 
        shuffle=False
    )
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            if is_hetero:
                logits = model(batch.x_dict, batch.edge_index_dict)[:batch['variant'].batch_size].squeeze()
                target = batch['variant'].y[:batch['variant'].batch_size]
            else:
                logits = model(batch.x, batch.edge_index)[:batch.batch_size].squeeze()
                target = batch.y[:batch.batch_size]
                
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


if __name__ == "__main__":
    print(f"Initializing compute environment on: {device}")
    
    hetero_data, homo_data, train_idx, test_idx = load_golden_tensors()
    run_xgboost(hetero_data, train_idx, test_idx)
    
    homo_loader = NeighborLoader(homo_data, num_neighbors=[15, 10], batch_size=1024, 
                                input_nodes=homo_data.train_mask, shuffle=True)
    hetero_loader = NeighborLoader(hetero_data, num_neighbors=[15, 10], batch_size=1024, 
                                  input_nodes=('variant', hetero_data['variant'].train_mask), shuffle=True)
    
    print("--- Running Baseline 2: GCN (Homogeneous) ---")
    gcn = HomogeneousGNN(in_channels=15, hidden_channels=64, out_channels=1, model_type="GCN").to(device)
    y_true, y_pred, y_prob = train_and_eval_gnn(gcn, homo_data, homo_loader, is_hetero=False)
    print_metrics("GCN", y_true, y_pred, y_prob)
    
    print("--- Running Baseline 3: GraphSAGE (Homogeneous) ---")
    sage = HomogeneousGNN(in_channels=15, hidden_channels=64, out_channels=1, model_type="GraphSAGE").to(device)
    y_true, y_pred, y_prob = train_and_eval_gnn(sage, homo_data, homo_loader, is_hetero=False)
    print_metrics("GraphSAGE", y_true, y_pred, y_prob)
    
    print("--- Running Proposed Architecture: GSTAT (Heterogeneous) ---")
    gstat = HeteroGSTAT(variant_dim=15, gene_dim=2, hidden_dim=64, heads=4, num_layers=3).to(device)
    y_true, y_pred, y_prob = train_and_eval_gnn(gstat, hetero_data, hetero_loader, is_hetero=True)
    print_metrics("GSTAT", y_true, y_pred, y_prob)