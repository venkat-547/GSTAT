import os
import csv
import sqlite3
import time

BASE_DIR = r"E:\Genomic"
META_CSV = os.path.join(BASE_DIR, "output", "variant_metadata.csv")
DB_PATH = os.path.join(BASE_DIR, "output", "search_index.db")

def build_index():
    print("Initiating SQLite index compilation with de-duplication...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE variants (
            tensor_idx INTEGER PRIMARY KEY,
            display_id TEXT,
            gene_symbol TEXT,
            clinical_significance TEXT,
            disease_name TEXT
        )
    ''')
    
    print("Streaming CSV to database and handling duplicates...")
    t0 = time.time()
    
    with open(META_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        batch = []
        for row in reader:
            batch.append((int(row[0]), row[1].lower(), row[2], row[3], row[4]))
            if len(batch) >= 100000:
                cursor.executemany('INSERT OR IGNORE INTO variants VALUES (?, ?, ?, ?, ?)', batch)
                batch = []
        if batch:
            cursor.executemany('INSERT OR IGNORE INTO variants VALUES (?, ?, ?, ?, ?)', batch)
            
    print("Creating B-Tree index on variant identifiers...")
    cursor.execute('CREATE INDEX idx_display_id ON variants(display_id)')
    
    conn.commit()
    conn.close()
    print(f"Database compilation completed in {time.time()-t0:.1f} seconds.")

if __name__ == "__main__":
    build_index()