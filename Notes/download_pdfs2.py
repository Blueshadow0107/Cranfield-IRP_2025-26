#!/usr/bin/env python3
"""Direct PDF download attempts with proper headers and redirects."""
import requests, os

OUTDIR = "Notes/papers"
os.makedirs(OUTDIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/pdf",
}

attempts = [
    # (filename, url)
    ("adamatzky2019", "https://arxiv.org/pdf/1811.09989.pdf"),  # preprint
    ("reyssat2009", "https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2009.0184"),
    ("egbert2018", "https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2018.0169"),
    ("egbert2019", "https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2019.0190"),
    ("adamatzky2002", "https://journals.aps.org/pre/pdf/10.1103/PhysRevE.66.046112"),
    ("prakash2007", "https://www.science.org/doi/pdf/10.1126/science.1134881"),
    ("weaver2010", "https://pubs.rsc.org/en/content/articlepdf/2010/lc/c004851b"),
    ("steinbock1996", "https://pubs.acs.org/doi/pdf/10.1021/jp962282p"),
    ("toth1995", "https://aip.scitation.org/doi/pdf/10.1063/1.470675"),
    ("baltussen2024", "https://www.nature.com/articles/s41586-024-07567-x.pdf"),
    ("nagipogu2025", "https://pubs.acs.org/doi/pdf/10.1021/acssynbio.5c00099"),
    ("stovold2016", "https://www.tandfonline.com/doi/pdf/10.1080/17445760.2016.1155579"),
    ("jaafar2022", "https://pubs.rsc.org/en/content/articlepdf/2022/nr/d2nr05012a"),
    ("zhang2024", "https://pubs.acs.org/doi/pdf/10.1021/acsami.3c19020"),
    ("ding2023", "https://pubs.rsc.org/en/content/articlepdf/2023/cs/d3cs00259d"),
    ("schroder2023", "https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-031822-041721"),
]

for name, url in attempts:
    path = os.path.join(OUTDIR, f"{name}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        print(f"[{name}] already exists ({os.path.getsize(path)} bytes)")
        continue
    try:
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"[{name}] downloaded ({len(r.content)} bytes)")
        else:
            print(f"[{name}] failed ({r.status_code}, {len(r.content)} bytes)")
    except Exception as e:
        print(f"[{name}] error: {e}")

print("\n--- Final files ---")
for f in sorted(os.listdir(OUTDIR)):
    fp = os.path.join(OUTDIR, f)
    print(f"  {f}: {os.path.getsize(fp)} bytes")
