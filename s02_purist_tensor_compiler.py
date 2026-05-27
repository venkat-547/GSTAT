import os
import sys
import time
import logging
import csv
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = r"E:\Genomic"
VCF_FILE = os.path.join(BASE_DIR, "data", "vcf", "clinvar_enriched.vcf")
REVEL_FILE = os.path.join(BASE_DIR, "tools", "REVEL", "revel_database.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROC_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

ACMG_GENES = {'BRCA1', 'BRCA2', 'TP53', 'APC', 'LDLR', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'MEN1', 'RET', 'PTEN', 'RB1', 'VHL', 'WT1', 'NF2', 'COL3A1', 'FBN1', 'TGFBR1', 'TGFBR2', 'SMAD3', 'ACTA2', 'MYH11', 'MYBPC3', 'MYH7', 'TNNT2', 'TNNI3', 'TPM1', 'MYL2', 'MYL3', 'PRKAG2', 'GLA', 'RYR2', 'PKP2', 'DSP', 'DSC2', 'TMEM43', 'DSG2', 'KCNQ1', 'KCNH2', 'SCN5A', 'LMNA', 'PCSK9', 'APOB', 'OTC', 'GAA', 'MUT', 'MMUT', 'ATP7B', 'HFE', 'SERPINA1', 'RYR1', 'CACNA1S', 'TSC1', 'TSC2', 'NF1', 'STK11', 'SDHD', 'SDHAF2', 'SDHC', 'SDHB'}
HIGH_PLI_GENES = {'ARID1B', 'ANKRD11', 'SCN1A', 'SCN2A', 'SCN8A', 'SYNGAP1', 'SHANK3', 'KMT2A', 'KMT2D', 'EP300', 'CREBBP', 'DYRK1A', 'GRIN2B', 'ADNP', 'CDH1', 'MECP2', 'FOXG1', 'CHD8', 'CHD2', 'SETD5', 'GATAD2B', 'TBL1XR1', 'MTOR', 'KIF1B', 'AGRN'}

def parse_info(info_str):
    info_dict = {}
    for item in info_str.split(';'):
        if '=' in item:
            k, v = item.split('=', 1)
            info_dict[k] = v
        else:
            info_dict[item] = True
    return info_dict

def extract_node_id(cols, info):
    if 'RS' in info: 
        return f"rs{info['RS']}"
    elif cols[2] != '.': 
        return f"clinvar:{cols[2]}"
    else: 
        return f"chr{cols[0]}:{cols[1]}"

def parse_snpeff_ann(ann_string):
    try:
        first_ann = ann_string.split(',')[0]
        impact = first_ann.split('|')[2].upper()
        return impact
    except IndexError:
        return "UNKNOWN"

def run_hetero_compiler():
    logging.info("Initiating heterogeneous tensor compilation")
    if not os.path.exists(VCF_FILE):
        logging.error(f"Target VCF not found at {VCF_FILE}.")
        sys.exit(1)
    if not os.path.exists(REVEL_FILE):
        logging.error(f"REVEL database not found at {REVEL_FILE}.")
        sys.exit(1)

    var_to_id, gene_to_id = {}, {}
    coord_to_vid = {}
    var_idx, gene_idx = 0, 0

    logging.info("Pass 1: Mapping heterogeneous nodes and spatial coordinates...")
    t0 = time.time()
    
    with open(VCF_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'): continue
            cols = line.strip().split('\t')
            if len(cols) < 8: continue
            
            info = parse_info(cols[7])
            if 'GENEINFO' not in info: continue
            
            display_id = extract_node_id(cols, info)
            gene = info['GENEINFO'].split(':')[0].upper()
            
            if display_id not in var_to_id:
                var_to_id[display_id] = var_idx
                
                chrom = f"chr{cols[0]}" if not cols[0].startswith('chr') else cols[0]
                coord_key = f"{chrom}:{cols[1]}:{cols[3]}:{cols[4]}"
                coord_to_vid[coord_key] = var_idx
                
                var_idx += 1
                
            if gene not in gene_to_id:
                gene_to_id[gene] = gene_idx
                gene_idx += 1

    logging.info(f"Pass 1 completed: {var_idx:,} variants mapped in {time.time()-t0:.1f}s")

    logging.info("Pre-allocating tensors...")
    x_var = torch.zeros((var_idx, 15), dtype=torch.float32)
    x_gene = torch.zeros((gene_idx, 2), dtype=torch.float32)
    y_var = torch.full((var_idx,), -1, dtype=torch.long)
    
    edges_var = torch.empty(var_idx, dtype=torch.long)
    edges_gene = torch.empty(var_idx, dtype=torch.long)

    logging.info("Pass 1.5: Integrating REVEL database using spatial coordinates...")
    t_rev = time.time()
    revel_hits = 0
    
    with open(REVEL_FILE, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        try:
            c_chr = header.index('chr')
            c_pos = header.index('grch38_pos')
            c_ref = header.index('ref')
            c_alt = header.index('alt')
            c_rev = header.index('REVEL')
        except ValueError:
            logging.error(f"REVEL CSV header structure unrecognized.")
            sys.exit(1)

        for line in f:
            cols = line.strip().split(',')
            if len(cols) < 8 or not cols[c_pos]: continue
            
            chrom = f"chr{cols[c_chr]}" if not cols[c_chr].startswith('chr') else cols[c_chr]
            coord_key = f"{chrom}:{cols[c_pos]}:{cols[c_ref]}:{cols[c_alt]}"
            
            if coord_key in coord_to_vid:
                v_id = coord_to_vid[coord_key]
                val = cols[c_rev]
                if val and val != '.':
                    current_val = x_var[v_id, 14].item()
                    new_val = float(val)
                    if new_val > current_val:
                        x_var[v_id, 14] = new_val
                        if current_val == 0.0:
                            revel_hits += 1

    logging.info(f"REVEL integration completed: {revel_hits:,} variants updated in {time.time()-t_rev:.1f}s")

    logging.info("Pass 2: Processing variant features and metadata...")
    t1 = time.time()
    
    meta_path = os.path.join(OUTPUT_DIR, "variant_metadata.csv")
    with open(meta_path, 'w', newline='', encoding='utf-8') as f_meta:
        meta_writer = csv.writer(f_meta)
        meta_writer.writerow(['tensor_idx', 'display_id', 'gene_symbol', 'clinical_significance', 'disease_name'])
        
        with open(VCF_FILE, 'r', encoding='utf-8') as f_vcf:
            for line in f_vcf:
                if line.startswith('#'): continue
                cols = line.strip().split('\t')
                if len(cols) < 8: continue
                
                info = parse_info(cols[7])
                if 'GENEINFO' not in info: continue
                
                display_id = extract_node_id(cols, info)
                v_id = var_to_id[display_id]
                gene_name = info['GENEINFO'].split(':')[0].upper()
                g_id = gene_to_id[gene_name]
                
                edges_var[v_id] = v_id
                edges_gene[v_id] = g_id
                
                mc_str = info.get('MC', '').lower()
                ann_str = info.get('ANN', '')
                
                if 'missense' in mc_str: 
                    x_var[v_id, 0] = 1.0
                elif 'synonymous' in mc_str: 
                    x_var[v_id, 1] = 1.0
                elif 'frameshift' in mc_str or 'nonsense' in mc_str: 
                    x_var[v_id, 2] = 1.0
                else: 
                    x_var[v_id, 3] = 1.0
                
                impact = parse_snpeff_ann(ann_str)
                if impact == 'HIGH': 
                    x_var[v_id, 4] = 1.0
                elif impact == 'MODERATE': 
                    x_var[v_id, 5] = 1.0
                elif impact == 'LOW': 
                    x_var[v_id, 6] = 1.0
                else: 
                    x_var[v_id, 7] = 1.0
                
                if 'AF_EXAC' in info:
                    af = float(info['AF_EXAC'])
                    x_var[v_id, 8] = -np.log10(af + 1e-9) if af > 0 else 0.0
                    x_var[v_id, 9] = 1.0
                else:
                    x_var[v_id, 8] = 0.0
                    x_var[v_id, 9] = 0.0
                
                is_acmg = 1.0 if gene_name in ACMG_GENES else 0.0
                is_pli = 1.0 if gene_name in HIGH_PLI_GENES else 0.0
                x_var[v_id, 10] = is_acmg
                x_var[v_id, 11] = is_pli
                
                revstat = info.get('CLNREVSTAT', '').lower()
                if 'exp' in revstat or 'mult' in revstat: 
                    x_var[v_id, 12] = 1.0
                elif 'single' in revstat: 
                    x_var[v_id, 12] = 0.5
                else: 
                    x_var[v_id, 12] = 0.1
                x_var[v_id, 13] = 1.0
                
                x_gene[g_id, 0] = is_acmg
                x_gene[g_id, 1] = is_pli
                
                clnsig = info.get('CLNSIG', '').lower()
                is_elite = x_var[v_id, 12] >= 0.5
                
                if is_elite and 'pathogenic' in clnsig and 'conflict' not in clnsig: 
                    y_var[v_id] = 1
                elif is_elite and 'benign' in clnsig and 'conflict' not in clnsig: 
                    y_var[v_id] = 0
                else: 
                    y_var[v_id] = -1

                clndn = info.get('CLNDN', 'Not_Provided').replace('%2C', ',').replace('_', ' ')
                meta_writer.writerow([v_id, display_id, gene_name, clnsig, clndn])

    logging.info(f"Pass 2 completed in {time.time()-t1:.1f}s")
    
    edge_index_var_gene = torch.stack([edges_var, edges_gene], dim=0)
    
    logging.info("Saving processed tensors to disk...")
    feature_dict = {'variant': x_var, 'gene': x_gene}
    label_dict = {'variant': y_var}
    edge_index_dict = {('variant', 'affects', 'gene'): edge_index_var_gene}
    
    torch.save(feature_dict, os.path.join(PROC_DIR, "hetero_features.pt"))
    torch.save(label_dict, os.path.join(PROC_DIR, "hetero_labels.pt"))
    torch.save(edge_index_dict, os.path.join(PROC_DIR, "hetero_edge_indices.pt"))

    logging.info("="*70)
    logging.info("Heterogeneous tensor compilation completed successfully.")
    logging.info(f"Variant features shape: {x_var.shape}")
    logging.info(f"Gene features shape: {x_gene.shape}")
    logging.info(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    run_hetero_compiler()