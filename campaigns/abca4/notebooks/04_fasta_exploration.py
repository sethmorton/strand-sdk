#!/usr/bin/env python3
"""Minimal Marimo app: show a FASTA file in a table."""

import marimo

app = marimo.App()

@app.cell
def __():
    import marimo as mo
    from Bio import SeqIO
    from pathlib import Path

    # Point this to your FASTA file
    fasta_path = Path("/home/ae25872/codebase/kleon/strand-sdk/campaigns/abca4/data_raw/sequences/ABCA4_P78363.fasta")

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

    # ⬇️ last expression is the cell output
    view


if __name__ == "__main__":
    app.run()
