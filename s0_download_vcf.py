import os
import gzip
import shutil
import time
import requests

DOWNLOAD_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz"
OUTPUT_DIR = os.path.dirname(__file__)
GZ_PATH = os.path.join(OUTPUT_DIR, "clinvar.vcf.gz")
VCF_PATH = os.path.join(OUTPUT_DIR, "clinvar.vcf")
CHUNK_SIZE = 8192


def stream_download(url, dest):
    print("Connecting to NCBI FTP server...")
    print(f"Source: {url}")
    print(f"Destination: {dest}")

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            t0 = time.time()

            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

                        if total and downloaded % (25 * 1024 * 1024) < CHUNK_SIZE:
                            pct = downloaded / total * 100
                            mb = downloaded / (1024 * 1024)
                            print(f"Progress: {mb:.0f} MB / {total/(1024*1024):.0f} MB ({pct:.1f}%)", end="\r")

        elapsed = time.time() - t0
        print(f"\nDownload completed in {elapsed:.1f}s ({os.path.getsize(dest)/(1024*1024):.1f} MB)")
        return True

    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return False


def decompress_gz(src, dest):
    print(f"Extracting {src} → {dest}")
    print("Decompression in progress...")

    try:
        t0 = time.time()
        with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"Decompression completed in {elapsed:.1f}s — File size: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"Decompression failed: {e}")
        return False


def main():
    print("=" * 65)
    print("G-STAT | Step 1: ClinVar VCF Acquisition")
    print("=" * 65)

    if os.path.exists(VCF_PATH):
        print(f"VCF file already exists at:\n   {VCF_PATH}")
        print("Delete the file if you want to download again.")
        return

    if not os.path.exists(GZ_PATH):
        success = stream_download(DOWNLOAD_URL, GZ_PATH)
        if not success:
            print("\nFailed to download the file.")
            print("Please check your internet connection and try again.")
            return
    else:
        print(f".gz archive already exists: {GZ_PATH}")

    success = decompress_gz(GZ_PATH, VCF_PATH)
    if not success:
        return

    print("Removing compressed archive to free disk space...")
    os.remove(GZ_PATH)

    print("\n" + "=" * 65)
    print("Download and extraction completed successfully.")
    print(f"Output file: {VCF_PATH}")
    print("Next: Upload this file to Galaxy for annotation (see hpc/galaxy_hpc_instructions.md)")
    print("=" * 65)


if __name__ == "__main__":
    main()