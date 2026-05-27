import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, LayerNorm, JumpingKnowledge


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
                ('variant', 'affects', 'gene'): GATv2Conv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim // heads, 
                    heads=heads, 
                    dropout=0.2, 
                    add_self_loops=False
                ),
                ('gene', 'rev_affects', 'variant'): GATv2Conv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim // heads, 
                    heads=heads, 
                    dropout=0.2, 
                    add_self_loops=False
                )
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
            h_residual_var = h_dict['variant']
            h_dict = self.convs[i](h_dict, edge_index_dict)
            
            h_dict['variant'] = F.gelu(self.norms_var[i](h_dict['variant']))
            h_dict['gene'] = F.gelu(self.norms_gene[i](h_dict['gene']))
            
            h_dict['variant'] = h_dict['variant'] + h_residual_var
            xs_variant.append(h_dict['variant'])
            
        h_var_fused = self.jump_var(xs_variant)
        h_var_final = F.gelu(self.jk_lin_var(h_var_fused))
        
        return self.classifier(h_var_final)