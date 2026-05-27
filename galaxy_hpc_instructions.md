# Galaxy HPC Instructions — Step 2: Enriching the ClinVar VCF

This guide walks you through uploading the `clinvar.vcf` file to Galaxy, performing functional annotation using SnpEff, adding REVEL scores via SnpSift, and downloading the final enriched file.

---

## What is Galaxy?

Galaxy[](https://usegalaxy.org) is a free, web-based scientific computing platform. It provides access to high-performance computing resources without requiring any local software installation.

---

## Part A — Create a Galaxy Account

1. Visit **https://usegalaxy.org**
2. Click **Login or Register** → **Register here**
3. Enter your email address, choose a password and username
4. Verify your email address
5. Log in to access the main analysis interface

---

## Part B — Upload Your VCF File

The raw file from Step 1 (`clinvar.vcf`) is approximately 1.76 GB.

**Recommended Method: Direct Upload**

1. Click the **Upload** button (cloud icon) in the top-left toolbar
2. Select **Choose local file**
3. Locate and select your `clinvar.vcf`
4. Set the datatype to `vcf`
5. Click **Start**
6. Once the upload completes, the file will appear in your History panel (right side)

**Note:** For very slow connections or larger files, you can use Galaxy’s FTP upload option.

---

## Part C — Run SnpEff Annotation

SnpEff annotates variants with their predicted functional impact.

1. In the tools panel on the left, search for **SnpEff eff**
2. Select **SnpEff eff: annotate variants**
3. Configure the tool as follows:

| Field                                   | Value                        |
|-----------------------------------------|------------------------------|
| Sequence changes (SNPs, MNPs, InDels)   | Select your `clinvar.vcf`    |
| Genome source                           | Named on demand              |
| SnpEff Genome Version                   | GRCh37.75                    |
| Output format                           | VCF                          |
| Create HTML report                      | No                           |
| Produce summary statistics              | No                           |

4. Click **Execute**

The job usually takes 2–6 hours. You can close the browser; the job will continue running. You will receive an email notification upon completion (if enabled).

5. Once the job completes successfully (turns green), rename the output dataset to `clinvar_snpeff.vcf`

---

## Part D — Add REVEL Scores with SnpSift

1. Search for **SnpSift dbNSFP** in the tools panel
2. Select **SnpSift dbNSFP: Annotate with dbNSFP**
3. Configure the following:

| Field                        | Value                              |
|------------------------------|------------------------------------|
| Variant input file           | Select `clinvar_snpeff.vcf`        |
| dbNSFP version               | Latest available (e.g., dbNSFP4.x) |
| Annotations to add           | REVEL_score, CADD_phred            |
| dbNSFP fields separator      | `,`                                |

4. Click **Execute**

This step typically takes 3–8 hours.

5. After completion, rename the output file to `clinvar_enriched.vcf`

---

## Part E — Download the Enriched VCF

The final enriched file is approximately 6.7 GB.

**Recommended: FTP Download**

1. Go to **User → Preferences → Manage Information** to view your FTP details
2. Use an FTP client such as FileZilla:
   - Host: `usegalaxy.org`
   - Port: `21`
   - Username & Password: your Galaxy login credentials
3. Download `clinvar_enriched.vcf`

**Alternative:** Direct download from the History panel (may be slow for large files).

---

## Part F — File Placement

Save the downloaded file to the following location:
E:\Genomic\data\vcf\clinvar_enriched.vcf
textIf you are using Linux or macOS, update the `BASE_DIR` variable in your Python scripts accordingly.

---

## Verification

To confirm the enrichment was successful, run:

```bash
python audit/s01_4_vcf_audit.py
You should see confirmation that REVEL scores and SnpEff annotations are present in the file.

Troubleshooting
| Problem                              | Solution |
|--------------------------------------|----------|
| SnpEff fails with "genome not found" | Install GRCh37.75 genome in SnpEff settings |
| Upload times out                     | Use FTP upload instead |
| REVEL scores are missing             | Check that correct dbNSFP version and fields were selected |
| Job stays queued for long time       | Shared queues may have delays during peak hours |
| Low REVEL fill rate                  | Ensure dbNSFP version is 4.x and field name is `REVEL_score` |

ProblemSolutionSnpEff fails with "genome not found"Install GRCh37.75 genome in SnpEff settingsUpload times outUse FTP upload insteadREVEL scores are missingCheck that correct dbNSFP version and fields were selectedJob stays queued for long timeShared queues may have delays during peak hoursLow REVEL fill rateEnsure dbNSFP version is 4.x and field name is REVEL_score

Expected Output
After successful annotation, each VCF line will contain additional fields like:
textANN=...|missense_variant|HIGH|BRCA1|...|p.Glu1138Gly;REVEL_score=0.854;CADD_phred=24.5
This clinvar_enriched.vcf file is the required input for the tensor compilation step (s02_purist_tensor_compiler.py).

You can now proceed to Step 3.
textThis version is clean, professional, easy to follow, and reads naturally like a well-written r