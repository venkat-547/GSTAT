import os
import sqlite3
import traceback
import torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from gstat_architecture import HeteroGSTAT

BASE_DIR = r"E:\Genomic"
PROC_DIR = os.path.join(BASE_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH = os.path.join(OUTPUT_DIR, "search_index.db")
MODEL_PATH = os.path.join(OUTPUT_DIR, "explicit_hetero_gstat_best.pth")

FEATURE_NAMES = [
    "Missense Map", "Synonymous Map", "Truncation Penalty", "Unknown Mut",
    "SnpEff HIGH", "SnpEff MODERATE", "SnpEff LOW", "SnpEff UNKNOWN",
    "ExAC Rareness AF", "AF Tracked", "ACMG Hub Gene", "High pLI Fragility",
    "ClinVar Elite Status", "Graph Constant", "REVEL Thermodynamics"
]

FEATURE_GLOSSARY = [
    {"head": "Sequence Ontology", "name": "Missense Map", "desc": "Mutation resulting in a different amino acid."},
    {"head": "Sequence Ontology", "name": "Synonymous Map", "desc": "DNA changes, but amino acid remains identical."},
    {"head": "Sequence Ontology", "name": "Truncation Penalty", "desc": "Destructive nonsense or frameshift mutations."},
    {"head": "Sequence Ontology", "name": "Unknown Mut", "desc": "Uncharacterized or intergenic structural changes."},
    {"head": "Structural Disruption", "name": "SnpEff HIGH", "desc": "Severe, high-impact structural disruption."},
    {"head": "Structural Disruption", "name": "SnpEff MODERATE", "desc": "Moderate-impact disruption (e.g., in-frame indel)."},
    {"head": "Structural Disruption", "name": "SnpEff LOW", "desc": "Low-impact variant (typically synonymous)."},
    {"head": "Structural Disruption", "name": "SnpEff UNKNOWN", "desc": "Structural impact could not be reliably determined."},
    {"head": "Population Genetics", "name": "ExAC Rareness AF", "desc": "Allele frequency in the healthy human population."},
    {"head": "Population Genetics", "name": "AF Tracked", "desc": "Confirms if Allele Frequency data was localized."},
    {"head": "Population Genetics", "name": "High pLI Fragility", "desc": "Probability of gene being Loss-of-Function Intolerant."},
    {"head": "Clinical Priors", "name": "ACMG Hub Gene", "desc": "Gene is a highly actionable clinical target."},
    {"head": "Clinical Priors", "name": "ClinVar Elite Status", "desc": "Variant has high-confidence clinical reviews."},
    {"head": "Topological Priors", "name": "Graph Constant", "desc": "Localized anchor bias for Neural Network attention."},
    {"head": "Atomic Physics", "name": "REVEL Thermodynamics", "desc": "Atomic-level thermodynamic destructiveness score."}
]

print("Loading neural architecture and data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features = torch.load(os.path.join(PROC_DIR, "hetero_features.pt"), weights_only=True)
edges = torch.load(os.path.join(PROC_DIR, "hetero_edge_indices.pt"), weights_only=True)
labels = torch.load(os.path.join(PROC_DIR, "hetero_labels.pt"), weights_only=True)

data = HeteroData()
data['variant'].x = features['variant']
data['gene'].x = features['gene']
data['variant'].y = labels['variant'].float().view(-1, 1)

fwd_edge = edges[('variant', 'affects', 'gene')]
data['variant', 'affects', 'gene'].edge_index = fwd_edge
data['gene', 'rev_affects', 'variant'].edge_index = torch.stack([fwd_edge[1], fwd_edge[0]], dim=0)

model = HeteroGSTAT(variant_dim=15, gene_dim=2, hidden_dim=64, heads=4, num_layers=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

print("Computing global database statistics...")
conn_init = sqlite3.connect(DB_PATH)
cursor_init = conn_init.cursor()

cursor_init.execute("SELECT COUNT(*) FROM variants")
total_global_variants = cursor_init.fetchone()[0]

cursor_init.execute("SELECT COUNT(DISTINCT gene_symbol) FROM variants")
total_global_genes = cursor_init.fetchone()[0]

cursor_init.execute("""
    SELECT COUNT(DISTINCT disease_name) 
    FROM variants 
    WHERE disease_name NOT LIKE '%not_provided%' 
      AND disease_name NOT LIKE '%not specified%' 
      AND disease_name NOT LIKE '%see cases%'
""")
total_global_diseases = cursor_init.fetchone()[0]

cursor_init.execute("""
    SELECT COUNT(*) FROM variants 
    WHERE clinical_significance LIKE '%benign%' 
      AND clinical_significance NOT LIKE '%conflicting%' 
      AND clinical_significance NOT LIKE '%pathogenic%'
""")
global_benign = cursor_init.fetchone()[0]

cursor_init.execute("""
    SELECT COUNT(*) FROM variants 
    WHERE clinical_significance LIKE '%pathogenic%' 
      AND clinical_significance NOT LIKE '%conflicting%' 
      AND clinical_significance NOT LIKE '%benign%'
""")
global_pathogen = cursor_init.fetchone()[0]

global_vus = total_global_variants - global_benign - global_pathogen
conn_init.close()
print("Global database audit completed.")

COLOR_TARGET = '#ff4d4d'
COLOR_GENE = '#66ff66'
COLOR_DISEASE = '#00d2ff'
COLOR_BIO = '#d946ef'
COLOR_PERIPHERAL = '#808080'
COLOR_DRIVER = '#ff6600'
COLOR_EDGE_GREY = '#d1d5db'

app = Dash(__name__)
app.title = "Genomic Analysis Report"

app.layout = html.Div(style={
    'font-family': '"Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    'backgroundColor': '#f3f4f6', 'color': '#111827', 'minHeight': '100vh',
    'margin': '0', 'padding': '0', 'display': 'flex', 'flexDirection': 'column'
}, children=[
    
    dcc.Store(id='global-stats-store', data={
        'benign': global_benign, 'pathogen': global_pathogen, 'vus': global_vus,
        'total': total_global_variants, 'genes': total_global_genes, 'diseases': total_global_diseases
    }),
    html.Div(id='dummy-anim-output', style={'display': 'none'}),
    
    html.Div([
        html.Div([
            html.H4("Global Database Metrics", style={'margin': '0 0 5px 0', 'color': '#1f2937', 'fontSize': '14px', 'fontWeight': 'bold'}),
            html.P([
                "Total Variants: ", html.Span("0", id="count-benign", style={'fontWeight': 'bold', 'color': '#059669'}), " Benign + ",
                html.Span("0", id="count-pathogen", style={'fontWeight': 'bold', 'color': '#dc2626'}), " Pathogenic + ",
                html.Span("0", id="count-vus", style={'fontWeight': 'bold', 'color': '#7c3aed'}), " VUS = ",
                html.Span("0", id="count-total", style={'fontWeight': 'bold'}), " Total"
            ], style={'margin': '2px 0', 'fontSize': '12px', 'color': '#4b5563'}),
            html.P(["Total Genes: ", html.Span("0", id="count-genes", style={'fontWeight': 'bold'})], style={'margin': '2px 0', 'fontSize': '12px', 'color': '#4b5563'}),
            html.P(["Total Diseases: ", html.Span("0", id="count-diseases", style={'fontWeight': 'bold'})], style={'margin': '2px 0', 'fontSize': '12px', 'color': '#4b5563'}),
        ], style={'position': 'absolute', 'top': '15px', 'left': '20px', 'textAlign': 'left', 'width': '320px'}),
        
        html.Div([
            html.H1("Genomic Analysis Report Engine", style={'textAlign': 'center', 'color': '#111827', 'margin': '0 0 5px 0', 'fontSize': '28px'}),
            dcc.Dropdown(
                id='variant-search', 
                placeholder='Query Variant ID (e.g., rs2100306794)...',
                style={'color': '#000000', 'fontSize': '16px', 'borderRadius': '4px'},
                searchable=True, 
                clearable=True
            )
        ], style={'width': '40%', 'margin': '0 auto'}),
        
        html.Div([
            html.H4("15-Dimensional Features", style={'margin': '0 0 5px 0', 'color': '#1f2937', 'fontSize': '14px', 'fontWeight': 'bold', 'paddingRight': '25px', 'textAlign': 'right'}),
            html.Div([
                html.Div(id='carousel-track', children=[
                    html.Div([
                        html.P(f.get('head'), style={'margin': '0', 'fontSize': '10px', 'color': '#f59e0b', 'textTransform': 'uppercase', 'fontWeight': 'bold'}),
                        html.P(f.get('name'), style={'margin': '1px 0', 'fontWeight': 'bold', 'fontSize': '13px', 'color': '#111827'}),
                        html.P(f.get('desc'), style={'margin': '0', 'fontSize': '11px', 'color': '#4b5563', 'lineHeight': '1.2'})
                    ], style={'width': f'{100/15}%', 'flexShrink': '0', 'padding': '0 25px', 'boxSizing': 'border-box', 'textAlign': 'right'})
                    for f in FEATURE_GLOSSARY
                ], style={'display': 'flex', 'width': '1500%', 'transition': 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)'})
            ], style={'overflow': 'hidden', 'width': '100%', 'position': 'relative'}),
            
            html.Button("<", id="carousel-btn-left", style={
                'position': 'absolute', 'top': '35px', 'left': '0px', 
                'background': 'none', 'border': 'none', 'fontSize': '20px', 
                'color': '#9ca3af', 'cursor': 'pointer', 'fontWeight': 'bold', 'padding': '0'
            }),
            html.Button(">", id="carousel-btn-right", style={
                'position': 'absolute', 'top': '35px', 'right': '0px', 
                'background': 'none', 'border': 'none', 'fontSize': '20px', 
                'color': '#9ca3af', 'cursor': 'pointer', 'fontWeight': 'bold', 'padding': '0'
            }),
            dcc.Interval(id='carousel-interval', interval=5000, n_intervals=0)
            
        ], style={'position': 'absolute', 'top': '15px', 'right': '20px', 'width': '300px'}),

    ], style={'padding': '15px 20px', 'backgroundColor': '#ffffff', 'borderBottom': '1px solid #d1d5db', 
              'boxShadow': '0 2px 4px rgba(0,0,0,0.05)', 'position': 'relative', 'minHeight': '90px', 
              'display': 'flex', 'alignItems': 'center'}),
    
    html.Div(id='diagnostic-output', style={
        'display': 'flex', 'flexDirection': 'row', 'flex': '1', 'padding': '20px', 
        'gap': '20px', 'height': 'calc(100vh - 120px)'
    })
])

app.clientside_callback(
    """
    function(data) {
        if(!data) return window.dash_clientside.no_update;
        const easeOutExpo = (x) => x === 1 ? 1 : 1 - Math.pow(2, -10 * x);
        const animateValue = (id, end, duration) => {
            let obj = document.getElementById(id);
            if(!obj) return;
            let startTimestamp = null;
            const step = (timestamp) => {
                if (!startTimestamp) startTimestamp = timestamp;
                let progress = Math.min((timestamp - startTimestamp) / duration, 1);
                let current = Math.floor(easeOutExpo(progress) * end);
                obj.innerHTML = current.toLocaleString(); 
                if (progress < 1) window.requestAnimationFrame(step);
                else obj.innerHTML = end.toLocaleString(); 
            };
            window.requestAnimationFrame(step);
        };
        setTimeout(() => {
            animateValue('count-benign', data.benign, 2000);
            animateValue('count-pathogen', data.pathogen, 2000);
            animateValue('count-vus', data.vus, 2000);
            animateValue('count-total', data.total, 2500); 
            animateValue('count-genes', data.genes, 2000);
            animateValue('count-diseases', data.diseases, 2000);
        }, 150);
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-anim-output', 'children'),
    Input('global-stats-store', 'data')
)

@app.callback(
    Output('carousel-track', 'style'),
    Input('carousel-interval', 'n_intervals'),
    Input('carousel-btn-right', 'n_clicks'),
    Input('carousel-btn-left', 'n_clicks')
)
def update_carousel(n_intervals, n_right, n_left):
    total_steps = (n_intervals or 0) + (n_right or 0) - (n_left or 0)
    idx = total_steps % 15
    return {
        'display': 'flex',
        'width': '1500%',
        'transition': 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        'transform': f'translateX(-{idx * (100/15)}%)'
    }

@app.callback(
    Output('variant-search', 'options'),
    Input('variant-search', 'search_value')
)
def update_search_options(search_value):
    if not search_value or len(search_value) < 2:
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT tensor_idx, display_id, clinical_significance 
                      FROM variants 
                      WHERE display_id LIKE ? LIMIT 15''', (f"{search_value.lower()}%",))
    results = cursor.fetchall()
    conn.close()
    return [{'label': f"{d_id.upper()} ({sig.upper() if sig != 'not_provided' else 'VUS'})", 
             'value': idx} for idx, d_id, sig in results]

@app.callback(
    Output('diagnostic-output', 'children'),
    Input('variant-search', 'value')
)
def run_inference(tensor_idx):
    if tensor_idx is None:
        return html.Div(html.H3("Awaiting target variant selection..."), 
                       style={'margin': 'auto', 'color': '#6b7280'})
    
    try:
        tensor_idx = int(tensor_idx)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT display_id, gene_symbol, disease_name, clinical_significance FROM variants WHERE tensor_idx = ?', (tensor_idx,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return html.Div(html.H3("Variant not found in database."), 
                           style={'margin': 'auto', 'color': '#ff4d4d'})
            
        d_id, g_sym, d_name, d_sig = row
        
        raw_disease = d_name.title() if d_name else ""
        is_missing = raw_disease.lower() in ["not_provided", "not specified", "see cases", ""]
        
        if is_missing:
            cursor.execute('''
                SELECT disease_name 
                FROM variants 
                WHERE gene_symbol = ? 
                  AND disease_name NOT LIKE '%not_provided%' 
                  AND disease_name NOT LIKE '%not specified%' 
                  AND disease_name NOT LIKE '%see cases%'
                GROUP BY disease_name 
                ORDER BY COUNT(disease_name) DESC 
                LIMIT 1
            ''', (g_sym,))
            inferred_row = cursor.fetchone()
            
            if inferred_row:
                graph_disease_text = inferred_row[0].title()
                ui_disease_text = html.Span([
                    f"{graph_disease_text} ",
                    html.Span(f"(Inferred via {g_sym} Knowledge Graph)", 
                             style={'color': '#f59e0b', 'fontSize': '12px', 'fontWeight': 'normal'})
                ])
            else:
                graph_disease_text = "Unknown Novel Condition"
                ui_disease_text = html.Span("Unknown (No gene-level mapping available)", 
                                          style={'color': '#ef4444'})
        else:
            graph_disease_text = raw_disease
            ui_disease_text = raw_disease

        loader = NeighborLoader(data, num_neighbors=[120, 15, 0], batch_size=1, 
                               input_nodes=('variant', torch.tensor([tensor_idx])))
        batch = next(iter(loader)).to(device)
        
        v_n_id = batch['variant'].n_id.cpu().numpy()
        peripheral_indices = [int(idx) for idx in v_n_id if idx != tensor_idx][:150]
        
        peripheral_metadata = {}
        if peripheral_indices:
            placeholders = ','.join(['?'] * len(peripheral_indices))
            cursor.execute(f'''
                SELECT tensor_idx, display_id, clinical_significance 
                FROM variants 
                WHERE tensor_idx IN ({placeholders})
            ''', tuple(peripheral_indices))
            peripheral_metadata = {r[0]: (r[1], r[2]) for r in cursor.fetchall()}
            
        conn.close()
        
        all_sigs = [d_sig.lower()] + [meta[1].lower() for meta in peripheral_metadata.values()]
        local_benign = local_pathogen = local_vus = 0
        
        for s in all_sigs:
            if 'conflicting' in s or ('benign' in s and 'pathogenic' in s):
                local_vus += 1
            elif 'pathogenic' in s:
                local_pathogen += 1
            elif 'benign' in s:
                local_benign += 1
            else:
                local_vus += 1
                
        total_local_variants = len(all_sigs)
        features_str = ", ".join(FEATURE_NAMES)
        
        model.eval()
        with torch.no_grad():
            baseline_logits = model(batch.x_dict, batch.edge_index_dict)[0]
            prob = torch.sigmoid(baseline_logits).item()
            pred_label = "PATHOGENIC" if prob > 0.2633 else "BENIGN"
            
            saliency_weights = np.zeros(15)
            original_features = batch.x_dict['variant'].clone()
            
            for i in range(15):
                masked_features = original_features.clone()
                masked_features[0, i] = 0.0
                batch.x_dict['variant'] = masked_features
                new_logits = model(batch.x_dict, batch.edge_index_dict)[0]
                saliency_weights[i] = abs(baseline_logits.item() - new_logits.item())
                
            batch.x_dict['variant'] = original_features

        sum_weights = np.sum(saliency_weights)
        saliency_normalized = (saliency_weights / sum_weights) * 100 if sum_weights > 0 else np.ones(15) * (100.0 / 15.0)
        
        G = nx.Graph()
        G.add_node("Disease", label=f"Condition:\n{graph_disease_text[:30]}...", 
                  hover=f"<b>Disease Mapping:</b><br>{graph_disease_text}", type='disease', 
                  color=COLOR_DISEASE, size=50)
        G.add_node("Target", label=f"Input:\n{d_id.upper()}", 
                  hover=f"<b>Target Variant:</b><br>{d_id.upper()}<br>Verdict: {pred_label}", 
                  type='target', color=COLOR_TARGET, size=45)
        G.add_node("Gene", label=f"Gene:\n{g_sym}", 
                  hover=f"<b>Associated Gene:</b><br>{g_sym}", type='gene', 
                  color=COLOR_GENE, size=45)
        
        G.add_node("Sig_Path", label=f"Pathogenic\nCluster: {local_pathogen}", 
                  hover=f"<b>Neighborhood Hostility:</b><br>{local_pathogen} known pathogenic mutations.", 
                  type='sig', color=COLOR_BIO, size=30)
        G.add_node("Sig_Benign", label=f"Benign\nCluster: {local_benign}", 
                  hover=f"<b>Neighborhood Tolerance:</b><br>{local_benign} known benign mutations.", 
                  type='sig', color=COLOR_BIO, size=30)
        
        G.add_edge("Target", "Gene", color=COLOR_DRIVER, width=2.5)
        G.add_edge("Gene", "Disease", color=COLOR_DRIVER, width=2.5)
        G.add_edge("Gene", "Sig_Path", color=COLOR_EDGE_GREY, width=1.5)
        G.add_edge("Gene", "Sig_Benign", color=COLOR_EDGE_GREY, width=1.5)
        
        for v_global in peripheral_indices:
            sis_name = f"Sis_{v_global}"
            if sis_name not in G:
                p_id, p_sig = peripheral_metadata.get(v_global, (f"Node_{v_global}", "Unknown"))
                hover_html = f"<b>Peripheral Variant: {p_id.upper()}</b><br>Clinical Status: {p_sig.replace('_', ' ').title()}"
                G.add_node(sis_name, label="", hover=hover_html, type='peripheral', 
                          color=COLOR_PERIPHERAL, size=28)
            G.add_edge(sis_name, "Gene", color=COLOR_EDGE_GREY, width=0.7)

        pos_2d = nx.spring_layout(G, dim=2, k=0.15, iterations=100)
        
        edge_traces = []
        for edge in G.edges():
            color = G.edges[edge].get('color', COLOR_EDGE_GREY)
            width = G.edges[edge].get('width', 1)
            x0, y0 = pos_2d[edge[0]]
            x1, y1 = pos_2d[edge[1]]
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], 
                                        mode='lines', line=dict(color=color, width=width), 
                                        hoverinfo='none', showlegend=False))
            
        node_x, node_y, node_col, node_size, node_text, node_hover = [], [], [], [], [], []
        for node in G.nodes():
            x, y = pos_2d[node]
            node_x.append(x)
            node_y.append(y)
            node_col.append(G.nodes[node]['color'])
            node_size.append(G.nodes[node]['size'])
            node_text.append(G.nodes[node]['label'])
            node_hover.append(G.nodes[node].get('hover', node))

        nodes_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
            hovertext=node_hover, hoverinfo='text', 
            textfont=dict(color='#000000', size=11, family="Arial"),
            marker=dict(size=node_size, color=node_col, opacity=0.9, line=dict(width=1, color='#ffffff')),
            showlegend=False
        )

        legend_elements = [
            ("Prediction Drivers", COLOR_DRIVER, "lines"),
            ("Input Variant", COLOR_TARGET, "markers"),
            ("Gene (Triplet)", COLOR_GENE, "markers"),
            ("Disease (Triplet)", COLOR_DISEASE, "markers"),
            ("Significance (Triplet)", COLOR_BIO, "markers"),
            ("Peripheral Variant", COLOR_PERIPHERAL, "markers")
        ]
        legend_traces = []
        for name, col, mod in legend_elements:
            if mod == "lines":
                legend_traces.append(go.Scatter(x=[None], y=[None], mode='lines', 
                                              line=dict(color=col, width=3), name=name))
            else:
                legend_traces.append(go.Scatter(x=[None], y=[None], mode='markers', 
                                              marker=dict(color=col, size=15), name=name))

        fig_graph = go.Figure(data=edge_traces + [nodes_trace] + legend_traces, 
                             layout=go.Layout(
                                 paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', 
                                 margin=dict(b=0, l=0, r=0, t=0),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 showlegend=True, 
                                 legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, 
                                            bgcolor="rgba(255,255,255,0.8)", font=dict(color="#000000"))
                             ))

        xai_pairs = [(f, w) for f, w in zip(FEATURE_NAMES, saliency_normalized)]
        xai_pairs.sort(key=lambda x: x[1], reverse=True)
        top_xai = xai_pairs[:7]
        top_xai.sort(key=lambda x: x[1])
        
        y_features = [x[0] for x in top_xai]
        max_val = max([x[1] for x in top_xai]) if top_xai else 1.0
        x_relative = [x[1] / max_val for x in top_xai] if max_val > 0 else [0] * len(top_xai)
        text_labels = [f"{x[1]:.1f}%" for x in top_xai]

        fig_xai = go.Figure(go.Bar(
            x=x_relative, y=y_features, orientation='h', text=text_labels, textposition='outside',
            textfont=dict(color='white', size=13, weight='bold'),
            marker=dict(color=x_relative, colorscale='YlOrRd', showscale=True, 
                       colorbar=dict(title=dict(text="Impact", font=dict(color='white')), 
                                    tickfont=dict(color='white')))
        ))
        
        fig_xai.update_layout(
            paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#ffffff'),
            margin=dict(l=10, r=20, t=40, b=10), 
            title=dict(text="Feature Importance:", font=dict(size=16, color='#ffffff')),
            xaxis=dict(title="Relative Impact", showgrid=False, tickfont=dict(color='white'), range=[0, 1.3]),
            yaxis=dict(title="", showgrid=False, tickfont=dict(color='white', size=12))
        )

        return [
            html.Div([
                html.H3("Network Topology Visualization", 
                       style={'color': '#000000', 'padding': '10px 0 0 20px', 'margin': '0', 
                              'backgroundColor': '#ffffff', 'fontWeight': 'normal'}),
                dcc.Graph(figure=fig_graph, style={'height': 'calc(100% - 110px)', 'width': '100%'}, 
                         config={'displayModeBar': False}),
                
                html.Div([
                    html.P(f"Local Node View: {local_benign} Benign + {local_pathogen} Pathogenic + {local_vus} VUS = {total_local_variants} Total Variants", 
                          style={'margin': '3px 0', 'fontWeight': 'bold'}),
                    html.P(f"Genes: 1", style={'margin': '3px 0', 'fontWeight': 'bold'}),
                    html.P(f"Diseases: 1", style={'margin': '3px 0', 'fontWeight': 'bold'}),
                    html.P(f"Feature Space: {features_str}", 
                          style={'margin': '3px 0', 'wordWrap': 'break-word', 'fontStyle': 'italic'})
                ], style={'padding': '10px 20px', 'backgroundColor': '#f8fafc', 'borderTop': '1px solid #e5e7eb', 
                         'fontSize': '13px', 'color': '#334155', 'height': '110px', 'boxSizing': 'border-box'})
                
            ], style={'width': '65%', 'backgroundColor': '#ffffff', 'borderRadius': '8px', 
                     'overflow': 'hidden', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 
                     'display': 'flex', 'flexDirection': 'column'}),
            
            html.Div([
                html.Div([
                    html.H3("Variant Analysis", style={'margin': '0 0 10px 0', 'color': '#ffffff', 'fontSize': '18px'}),
                    html.P(f"Variant RSID: {d_id.upper()}", style={'margin': '5px 0', 'color': '#d1d5db', 'fontSize': '14px'}),
                    html.P(f"Associated Gene: {g_sym}", style={'margin': '5px 0', 'color': '#d1d5db', 'fontSize': '14px'}),
                    html.P(["Associated Disease Risk: ", ui_disease_text], style={'margin': '5px 0', 'color': '#d1d5db', 'fontSize': '14px'}),
                    html.P(f"AI Confidence: {prob:.2%}", style={'margin': '5px 0', 'color': '#00f0ff', 'fontSize': '14px', 'fontWeight': 'bold'}),
                    html.P(f"AI Verdict: {pred_label}", style={'margin': '10px 0 0 0', 
                            'color': '#ff4d4d' if pred_label=="PATHOGENIC" else '#00e676', 'fontWeight': 'bold'}),
                ], style={'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'marginBottom': '15px'}),
                
                html.Div([
                    dcc.Graph(figure=fig_xai, style={'height': '100%', 'width': '100%'}, 
                             config={'displayModeBar': False})
                ], style={'flex': '1', 'backgroundColor': '#000000', 'borderRadius': '8px', 'overflow': 'hidden'})
                
            ], style={'width': '35%', 'display': 'flex', 'flexDirection': 'column'})
        ]

    except Exception:
        err_msg = traceback.format_exc()
        return html.Div([
            html.H2("System Error", style={'color': '#ff4d4d', 'margin': '0 0 10px 0'}),
            html.P("An error occurred during execution:", style={'color': '#ffffff'}),
            html.Pre(err_msg, style={'color': '#00ff00', 'backgroundColor': '#000000', 
                                   'padding': '20px', 'borderRadius': '8px', 'overflowX': 'auto'})
        ], style={'width': '100%', 'padding': '20px', 'backgroundColor': '#111827', 'borderRadius': '8px'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)