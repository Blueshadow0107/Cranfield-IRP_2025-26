#!/usr/bin/env python3
"""Download open-access PDFs for lit-review papers via Unpaywall."""
import requests, os, time, json

OUTDIR = "Notes/papers"
EMAIL = "user@example.com"  # required by Unpaywall API

PAPERS = [
    ("jaafar2022", "10.1039/D2NR05012A"),
    ("zhang2024", "10.1021/acsami.3c19020"),
    ("ding2023", "10.1039/D3CS00259D"),
    ("stovold2016", "10.1080/17445760.2016.1155579"),
    ("baltussen2024", "10.1038/s41586-024-07567-x"),
    ("egbert2018", "10.1098/rsif.2018.0169"),
    ("egbert2019", "10.1098/rsif.2019.0190"),
    ("nagipogu2025", "10.1021/acssynbio.5c00099"),
    ("prakash2007", "10.1126/science.1134881"),
    ("weaver2010", "10.1039/C004851B"),
    ("kim2023", "10.1038/s41467-023-42885-0"),
    ("adamatzky2002", "10.1103/PhysRevE.66.046112"),
    ("toth1995", "10.1063/1.470675"),
    ("steinbock1996", "10.1021/jp962282p"),
    ("adamatzky2019", "10.1098/rstb.2018.0372"),
    ("reyssat2009", "10.1098/rsif.2009.0184"),
    ("schroder2023", "10.1146/annurev-fluid-031822-041721"),
]

def fetch_unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Unpaywall error: {e}")
    return None

def download_pdf(url, path):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"  Download error: {e}")
    return False

os.makedirs(OUTDIR, exist_ok=True)
results = []

for name, doi in PAPERS:
    print(f"[{name}] DOI: {doi}")
    pdf_path = os.path.join(OUTDIR, f"{name}.pdf")
    if os.path.exists(pdf_path):
        print(f"  Already exists ({os.path.getsize(pdf_path)} bytes)")
        results.append((name, "exists"))
        continue

    data = fetch_unpaywall(doi)
    if not data:
        print("  Unpaywall lookup failed")
        results.append((name, "unpaywall_failed"))
        time.sleep(1)
        continue

    pdf_url = None
    # Try best OA location first
    if data.get("best_oa_location") and data["best_oa_location"].get("url_for_pdf"):
        pdf_url = data["best_oa_location"]["url_for_pdf"]
    else:
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                pdf_url = loc["url_for_pdf"]
                break

    if not pdf_url:
        print(f"  No OA PDF found (is_oa={data.get('is_oa')})")
        results.append((name, "no_oa"))
        time.sleep(1)
        continue

    print(f"  Downloading from: {pdf_url}")
    if download_pdf(pdf_url, pdf_path):
        print(f"  Saved ({os.path.getsize(pdf_path)} bytes)")
        results.append((name, "downloaded"))
    else:
        print("  Download failed")
        results.append((name, "dl_failed"))
    time.sleep(1.5)

print("\n--- SUMMARY ---")
for name, status in results:
    print(f"{name}: {status}")
