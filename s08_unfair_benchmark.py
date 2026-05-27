import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, LayerNorm, JumpingKnowledge
from torch_geometric.loader import NeighborLoader


class GatedHeteroProjection(nn.Module):
    def __init__(self, var_dim, gene_dim, hidden_dim):
        super().__init__()
        self.var_lin1 = nn.Linear(var_dim, hidden_dim)
        self.var_lin2 = nn.Linear(var_dim, hidden_dim)
        self.gene_lin1 = nn.Linear(gene_dim, hidden_dim)
        self.gene_lin2 = nn.Linear(gene_dim, hidden_dim)

    def forward(self, x_dict):
        h_dict = {}
        h_dict['variant'] = self.var_lin1(x_dict['variant']) * torch.sigmoid(self.var_lin2(x_dict['variant']))
        h_dict['gene'] = self.gene_lin1(x_dict['gene']) * torch.sigmoid(self.gene_lin2(x_dict['gene']))
        return h_dict


class HeteroGSTAT(nn.Module):
    def __init__(self, variant_dim=15, gene_dim=2, hidden_dim=64, heads=4, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.projection = GatedHeteroProjection(variant_dim, gene_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms_var = nn.ModuleList()
        self.norms_gene = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                ('variant', 'affects', 'gene'): GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, add_self_loops=False),
                ('gene', 'rev_affects', 'variant'): GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, add_self_loops=False)
            }, aggr='mean')
            self.convs.append(conv)
            self.norms_var.append(LayerNorm(hidden_dim))
            self.norms_gene.append(LayerNorm(hidden_dim))
        
        self.jump_var = JumpingKnowledge(mode='cat')
        self.jk_lin_var = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.BatchNorm1d(hidden_dim // 2), 
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        h_dict = self.projection(x_dict)
        xs_variant = []
        
        for i in range(self.num_layers):
            h_res = h_dict['variant']
            h_dict = self.convs[i](h_dict, edge_index_dict)
            h_dict['variant'] = F.gelu(self.norms_var[i](h_dict['variant'])) + h_res
            h_dict['gene'] = F.gelu(self.norms_gene[i](h_dict['gene']))
            xs_variant.append(h_dict['variant'])
        
        h_fused = self.jump_var(xs_variant)
        return self.classifier(F.gelu(self.jk_lin_var(h_fused)))


def run_benchmark():
    BASE_DIR = r"E:\Genomic"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing benchmark on device: {device}")
    
    print("Loading tensors...")
    features = torch.load(os.path.join(BASE_DIR, "processed", "hetero_features.pt"), weights_only=False)
    edges = torch.load(os.path.join(BASE_DIR, "processed", "hetero_edge_indices.pt"), weights_only=False)
    labels = torch.load(os.path.join(BASE_DIR, "processed", "hetero_labels.pt"), weights_only=False)
    
    data = HeteroData()
    data['variant'].x = features['variant']
    data['gene'].x = features['gene']
    data['variant'].y = labels['variant'].float()
    
    fwd_edge = edges[('variant', 'affects', 'gene')]
    data['variant', 'affects', 'gene'].edge_index = fwd_edge
    data['gene', 'rev_affects', 'variant'].edge_index = torch.stack([fwd_edge[1], fwd_edge[0]], dim=0)

    valid_indices = (data['variant'].y != -1).nonzero(as_tuple=True)[0]
    print(f"Found {len(valid_indices):,} valid ClinVar targets for benchmarking.")

    loader = NeighborLoader(
        data,
        num_neighbors=[30, 15, 10],
        batch_size=1024,
        input_nodes=('variant', valid_indices),
        shuffle=False
    )
    
    print("Loading model weights...")
    model = HeteroGSTAT().to(device)
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "output", "explicit_hetero_gstat_best.pth"), 
        map_location=device, 
        weights_only=True
    ))
    model.eval()

    y_true_list = []
    y_gstat_list = []
    y_revel_list = []

    print("Running inference on graph batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            
            logits = model(batch.x_dict, batch.edge_index_dict)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            bs = batch['variant'].batch_size
            
            y_gstat_list.extend(probs[:bs])
            y_true_list.extend(batch['variant'].y[:bs].cpu().numpy().flatten())
            y_revel_list.extend(batch['variant'].x[:bs, 14].cpu().numpy().flatten())
            
            if i % 10 == 0 and i > 0:
                print(f"Processed {i * 1024:,} / {len(valid_indices):,} targets...")

    print("Calculating final metrics...")
    fpr_g, tpr_g, _ = roc_curve(y_true_list, y_gstat_list)
    auc_g = auc(fpr_g, tpr_g)
    
    fpr_r, tpr_r, _ = roc_curve(y_true_list, y_revel_list)
    auc_r = auc(fpr_r, tpr_r)

    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"G-STAT AUROC : {auc_g:.4f}")
    print(f"REVEL AUROC  : {auc_r:.4f}")
    print("="*50)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_g, tpr_g, label=f'G-STAT (AUC = {auc_g:.3f})', color='red', lw=2)
    plt.plot(fpr_r, tpr_r, label=f'REVEL (AUC = {auc_r:.3f})', color='blue', lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Benchmark: G-STAT vs REVEL')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    
    plot_path = os.path.join(BASE_DIR, "output", "SOTA_ROC.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {plot_path}")


if __name__ == "__main__":
    run_benchmark()