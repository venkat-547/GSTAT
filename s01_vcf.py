import os
import sys
import time
import gzip
import shutil
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = r"E:\Genomic"
VCF_DIR = os.path.join(BASE_DIR, "data", "vcf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VCF_FILE = os.path.join(VCF_DIR, "clinvar.vcf")
VCF_GZ_FILE = os.path.join(VCF_DIR, "clinvar.vcf.gz")

VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"

def ensure_directories():
    os.makedirs(VCF_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Verified directory structure at {VCF_DIR}")

def stream_download_vcf() -> bool:
    if os.path.exists(VCF_FILE):
        logging.info(f"Uncompressed VCF already exists at {VCF_FILE}. Skipping download.")
        return True

    logging.info("Initiating secure VCF streaming download")
    logging.info(f"   Source: {VCF_URL}")
    logging.info(f"   Destination: {VCF_GZ_FILE}")

    start_time = time.time()
    
    try:
        with requests.get(VCF_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(VCF_GZ_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0 and downloaded_size % (1024 * 1024 * 25) < 8192:
                            mb_down = downloaded_size / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            pct = (downloaded_size / total_size) * 100
                            print(f"     -> Network Stream: {mb_down:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)...", end='\r')

        duration = time.time() - start_time
        print()
        logging.info(f"Secure download completed in {duration:.1f} seconds.")
        return True
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {e}")
        return False

def stream_decompress_vcf() -> bool:
    logging.info("Starting streaming decompression")
    
    if not os.path.exists(VCF_GZ_FILE):
        logging.error("Compressed archive not found for decompression.")
        return False
        
    start_time = time.time()
    
    try:
        with gzip.open(VCF_GZ_FILE, 'rb') as f_in:
            with open(VCF_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024*1024)
                
        duration = time.time() - start_time
        logging.info(f"Decompression completed in {duration:.1f} seconds.")
        
        logging.info("Performing disk cleanup - removing compressed archive")
        os.remove(VCF_GZ_FILE)
        
        vcf_size_mb = os.path.getsize(VCF_FILE) / (1024 * 1024)
        logging.info(f"Final VCF ready: {vcf_size_mb:.1f} MB at {VCF_FILE}")
        return True
        
    except Exception as e:
        logging.error(f"Decompression error: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("G-STAT MASTER PIPELINE | PHASE 1.2: VCF ACQUISITION")
    print("="*70)
    
    ensure_directories()
    
    if stream_download_vcf():
        if not os.path.exists(VCF_FILE):
            stream_decompress_vcf()
        else:
            logging.info("VCF is already decompressed and ready.")