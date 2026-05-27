# Galaxy HPC Instructions — Step 2: Enriching the ClinVar VCF

This document explains exactly how to upload `clinvar.vcf` to Galaxy, run SnpEff and REVEL annotation, and download the enriched output.

---

## What Is Galaxy?

Galaxy (https://usegalaxy.org) is a free, browser-based High-Performance Computing (HPC) platform. You do not need to install anything locally. All annotation jobs run on their distributed cluster.

---

## Part A — Create a Galaxy Account

1. Go to **https://usegalaxy.org**
2. Click **Login or Register** → **Register here**
3. Fill in your email, password, and username
4. Confirm your email address
5. Log in — you will land on the main analysis interface

---

## Part B — Upload Your VCF to Galaxy

Your file from Step 1 is `clinvar.vcf` (~1.76 GB).

**Method 1 — Direct Upload (files under 2 GB):**

1. In the Galaxy toolbar, click the **Upload** button (cloud icon, top-left of the tool panel)
2. Click **Choose local file**
3. Navigate to and select your `clinvar.vcf`
4. In the **Type** column, type or select: `vcf`
5. Click **Start**, then **Close** when the upload finishes
6. The file will appear in your **History** panel on the right as a green block

**Method 2 — FTP Upload (for files over 2 GB or slow connections):**

> The enriched VCF from this step will be ~6.7 GB, so you will need FTP for the download. For the initial 1.76 GB upload, direct upload usually works fine.

---

## Part C — Run SnpEff Annotation

SnpEff determines the structural consequence of each variant (missense, frameshift, stop gained, etc.) by cross-referencing the Ensembl gene database.

**Step-by-step:**

1. In the left tool panel, search for: **SnpEff eff**
2. Click **SnpEff eff: annotate variants** to open the tool form
3. Configure the fields as follows:

| Field | Value |
|---|---|
| **Sequence changes (SNPs, MNPs, InDels)** | Select your uploaded `clinvar.vcf` |
| **Genome source** | Named on demand |
| **SnpEff Genome Version** | `GRCh37.75` |
| **Output format** | `VCF (only if input is VCF)` |
| **Create HTML report** | No (saves time) |
| **Produce summary statistics** | No |
| **Filter out variants with upstream/downstream effects** | No |
| **Annotations to include** | Leave as default |

4. Click **Execute**

> ⏱ **Runtime:** 2–6 hours on the shared cluster. You can close your browser — the job continues running. You will receive an email when it finishes (if you set this up in Galaxy preferences).

5. When the job turns **green**, click it and rename the output to `clinvar_snpeff.vcf` for clarity

---

## Part D — Download dbNSFP and Run SnpSift for REVEL Scores

REVEL scores require the dbNSFP database. Galaxy hosts a pre-built version via SnpSift.

1. In the tool panel, search for: **SnpSift dbNSFP**
2. Click **SnpSift dbNSFP: Annotate with dbNSFP**
3. Configure:

| Field | Value |
|---|---|
| **Variant input file in VCF format** | Select `clinvar_snpeff.vcf` (output from Part C) |
| **dbNSFP version** | Select the newest available (e.g., `dbNSFP4.x`) |
| **Annotations to add** | Select: `REVEL_score`, `CADD_phred` |
| **dbNSFP fields separator** | `,` |

4. Click **Execute**

> ⏱ **Runtime:** 3–8 hours. This is the most time-consuming step.

5. When complete, rename the output to `clinvar_enriched.vcf`

---

## Part E — Download the Enriched VCF

The enriched file will be approximately **6.7 GB**. Use the FTP method for a reliable download.

**Galaxy FTP Download:**

1. In Galaxy, go to **User → Preferences → Manage Information**
2. Note your FTP upload/download folder path
3. In your History, click the **Download** icon on `clinvar_enriched.vcf`
4. Choose **Download dataset** → a direct link will appear, or
5. Use a proper FTP client like **FileZilla** (free) with:
   - **Host:** `usegalaxy.org`
   - **Port:** `21`
   - **Username / Password:** your Galaxy credentials

**Alternatively, direct browser download:**

1. Click the dataset in your History (green block)
2. Click the **Save icon** (floppy disk)
3. The file downloads as a `.vcf` — this works fine for files under ~4 GB on fast connections

---

## Part F — Place the File in the Correct Location

Move the downloaded file to:

```
E:\Genomic\data\vcf\clinvar_enriched.vcf
```

If you are on Linux/Mac, update `BASE_DIR` in `src/s02_purist_tensor_compiler.py`:

```python
BASE_DIR = "/your/path/to/genomic"
```

---

## Verification — Check the Enrichment Worked

Run the physics audit to confirm REVEL scores and SnpEff tags are present:

```bash
python audit/s01_4_vcf_audit.py
```

You should see output like:
```
✅ HEADER CHECK : PASSED. Physics tags detected in metadata.
• REVEL_score    : Found in  XX.X% of variants.
• CADD_phred     : Found in  XX.X% of variants.
✅ VALIDATION SUCCESS: Biochemical features are present and ready for tensor extraction.
```

If you see `❌ CRITICAL FAILURE`, repeat Part D — the SnpSift step may not have completed correctly.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| SnpEff job fails with "genome not found" | Manually install: In SnpEff tool settings, run a genome installation step for `GRCh37.75` first |
| Upload times out for large files | Use Galaxy's FTP uploader instead of the web interface |
| dbNSFP tool not available | Search for "SnpSift" instead — the REVEL annotation tool may be listed differently on your Galaxy instance |
| Job stuck in grey/queued state | Galaxy shared queues can take up to 24 h during peak times; this is normal |
| REVEL fill rate is 0% | The REVEL tool requires `dbNSFP` version 2.9+ and the field name must exactly match `REVEL_score` |

---

## What the HPC Pipeline Produces

After both SnpEff and SnpSift, each VCF row that was previously just coordinates will contain:

```
Before enrichment:
17  43094892  .  G  A  .  .  CLNSIG=Pathogenic;CLNDN=Breast_cancer

After enrichment:
17  43094892  .  G  A  .  .  CLNSIG=Pathogenic;CLNDN=Breast_cancer;
ANN=A|missense_variant|HIGH|BRCA1|...|p.Glu1138Gly;
REVEL_score=0.854;CADD_phred=24.5
```

This enriched file is the input to `src/s02_purist_tensor_compiler.py` in Step 3.
