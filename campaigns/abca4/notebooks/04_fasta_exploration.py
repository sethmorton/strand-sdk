#!/usr/bin/env python3
"""Minimal Marimo app: show a FASTA file in a table."""

import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def _():
    import re

    def read_fasta_sequence(path):
        """Read the first sequence from a FASTA file and return it as a single string."""
        seq_lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    # header line – skip for now
                    continue
                seq_lines.append(line)
        return "".join(seq_lines)


    def find_motifs(seq):
        """
        Find:
        - ATP-binding P-loop/Walker A–like motifs: GxxxxGK[ST]
        - N-glycosylation motifs: N{P}[ST]{P}
        Returns dict with lists of hits.
        """

        # Walker A / P-loop motif (ATP-binding):
        # GxxxxGKST is canonical; we’ll allow S or T in the last position: GxxxxGK[ST]
        atp_pattern = re.compile(r"G....GK[ST]")

        # N-glycosylation motif:
        # N{P}[ST]{P}  -> N-X-S/T, where X != P and last != P
        nglyc_pattern = re.compile(r"N[^P][ST][^P]")

        atp_sites = []
        for m in atp_pattern.finditer(seq):
            start = m.start() + 1  # 1-based indexing
            end = m.end()
            atp_sites.append({
                "motif": m.group(),
                "start": start,
                "end": end,
            })

        nglyc_sites = []
        for m in nglyc_pattern.finditer(seq):
            start = m.start() + 1
            end = m.end()
            nglyc_sites.append({
                "motif": m.group(),
                "start": start,
                "end": end,
            })
        FASTA_PATH = "strand-sdk/campaigns/abca4/data_raw/sequences/ABCA4_P78363.fasta"   # <-- put your FASTA file path here
        seq = read_fasta_sequence(FASTA_PATH)
        motifs = find_motifs(seq)

        print("ATP-binding-like motifs (GxxxxGK[ST]):")
        for site in motifs["atp_sites"]:
            print(f"  {site['motif']}  at {site['start']}-{site['end']}")

        print("\nN-glycosylation motifs (N[^P][ST][^P]):")
        for site in motifs["nglyc_sites"]:
            print(f"  {site['motif']}  at {site['start']}-{site['end']}")

        return {"atp_sites": atp_sites, "nglyc_sites": nglyc_sites}


@app.cell
def _():
    import marimo as mo
    from Bio import SeqIO
    from pathlib import Path

    FASTA_PATH = "strand-sdk/campaigns/abca4/data_raw/sequences/ABCA4_P78363.fasta"


    # Point this to your FASTA file
    fasta_path = Path(FASTA_PATH)

    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        view = mo.md(f"⚠️ FASTA file not found at `{fasta_path}`")
    else:
        if not records:
            view = mo.md("No sequences found in FASTA file.")
        else:

            rows = [
                {
                    "id": r.id,
                    "description": r.description,
                    "length": len(r.seq),
                    "sequence": str(r.seq),
                }
                for r in records
            ]
            view = mo.ui.table(rows)
            with open(fasta_path, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        print(line)
                    else:
                        print(line)

    print(view)
    return view

if __name__ == "__main__":
    app.run()
