use anyhow::Result;
use std::io::{self, Write};

fn print_lines(lines: &[&str]) -> Result<()> {
    let mut out = io::stdout().lock();
    for l in lines {
        writeln!(out, "{}", l)?;
    }
    Ok(())
}

pub fn debug(path: &str) -> Result<()> {
    let lines = vec![
        &*format!("macbook:~/geneloop-demo $ geneloop debug {}", path),
        "[GeneLoop] Scanning filesystem (cwd, neighbors)… ok",
        "[GeneLoop] Detecting file type…      ok (BAM)",
        "[GeneLoop] Quick sanity checks…      ok (sorted: coordinate)",
        "[GeneLoop] Index presence…           missing (.bai not found)",
        "[GeneLoop] Read summary (fast scan):",
        "  - total reads: ~42,317,894",
        "  - mapped: 97.6%   | properly paired: 95.8%",
        "  - duplicates: 11.9% (flag-based)",
        "  - multimappers: 6.3%",
        "  - median insert size: 286 bp",
        "  - GC content (reads): 42% | reference: 41%",
        "[Model] Thinking…",
        "[Model] Hypothesis: BAM is sorted; missing sidecar index (.bai).",
        "[Model] Evidence: SO:coordinate in header; .bai absent; typical idxstats/IGV failure.",
        "[Model] Plan: propose samtools index; re-check; warn on multimappers/GC.",
        "[Issue 1] Missing BAM index",
        "  Fix: samtools index -@ 8 sample.bam",
        "[Issue 2] Multimapper noise (6.3%) — consider MAPQ >=10 or STAR",
        "[Note] GC bias (coarse) — no strong deviation observed",
        "[GeneLoop] Suggested next command:",
        "  samtools index -@ 8 sample.bam && samtools idxstats sample.bam | head",
        "[Decision] Auto-generate fix command? [y/N]: y",
        "Running: samtools index -@ 8 sample.bam",
        "  writing index… done",
        "[Index] Done: sample.bam.bai (2.1s)",
        "Re-checking…",
        "[GeneLoop] Loading index… ok",
        "[GeneLoop] Index presence…           ok",
        "[GeneLoop] Ready for IGV / variant calling.",
        "Summary",
        "  Root cause: Missing .bai after sort",
        "  Fix applied: samtools index -@ 8 sample.bam",
        "  Risk notes: multimappers ~6.3%; consider MAPQ filter (>=10)",
    ];
    print_lines(&lines)
}

pub fn gen(spec: &str) -> Result<()> {
    let lines = vec![
        &*format!("macbook:~/geneloop-demo $ geneloop gen \"{}\"", spec),
        "[GeneLoop] Parsing intent… ok",
        "[Model] Reasoning: paired-end RNA‑seq; hg38; prefer bwa-mem2 for demo speed.",
        "[Model] Plan: emit minimal idempotent slice + environment pins.",
        "[GeneLoop] Resolving toolchain (bwa-mem2, samtools)… ok",
        "[GeneLoop] Emitting idempotent Snakemake slice (FASTQ ➜ BAM)",
        "# Snakefile",
        "rule all:",
        "    input: expand(\"results/bam/{sample}.bam.bai\", sample=lambda: config['samples'])",
        "",
        "rule align_and_sort:",
        "    input:",
        "        r1=\"data/fastq/{sample}_R1.fastq.gz\",",
        "        r2=\"data/fastq/{sample}_R2.fastq.gz\",",
        "        ref=\"refs/hg38.fa\"",
        "    output:",
        "        bam=temp(\"results/bam/{sample}.bam\"),",
        "        bai=\"results/bam/{sample}.bam.bai\"",
        "    params:",
        "        threads=8,",
        "        bwa_opts=\"-R '@RG\\tID:{wildcards.sample}\\tSM:{wildcards.sample}\\tPL:ILLUMINA'\"",
        "    log: \"logs/{sample}.align.log\"",
        "    conda: \"envs/aln.yaml\"",
        "    shell:",
        "        r\"\"\"",
        "        set -eo pipefail",
        "        if [ ! -s {output.bam} ]; then",
        "          bwa-mem2 mem -t {params.threads} {params.bwa_opts} {input.ref} {input.r1} {input.r2} 2>> {log} | \\",
        "          samtools view -@ {params.threads} -bS - | \\",
        "          samtools sort -@ {params.threads} -o {output.bam} - 2>> {log}",
        "        fi",
        "        samtools index -@ {params.threads} {output.bam} 2>> {log}",
        "        \"\"\"",
        "",
        "# envs/aln.yaml",
        "name: biocode-aln",
        "channels: [conda-forge, bioconda]",
        "dependencies:",
        "  - bwa-mem2=2.*",
        "  - samtools=1.19.*",
        "",
        "# config.yaml",
        "samples:",
        "  - sampleA",
        "  - sampleB",
        "[GeneLoop] Files written: Snakefile, envs/aln.yaml, config.yaml (virtual).",
        "Notes: Replace BWA-MEM2 with STAR for spliced RNA‑seq if preferred.",
        "Idempotent: no re-run if outputs exist; index enforced.",
    ];
    print_lines(&lines)
}

pub fn explain(path: &str) -> Result<()> {
    let lines = vec![
        &*format!("macbook:~/geneloop-demo $ geneloop explain {}", path),
        &*format!("[GeneLoop] Reading file: {}…", path),
        "[Explain] Summary",
        "  - Purpose: parse FASTA records and print id\tlength TSV.",
        "  - Flow: detect headers (\">id\"), accumulate sequence lines per id, output lengths.",
        "[Explain] Assumptions",
        "  - Header regex: /^(\\\\S+)/ captures ID without spaces.",
        "  - Multiline sequences supported; blank lines ignored.",
        "[Explain] Pitfalls",
        "  - Memory growth: stores entire sequences in a hash (large FASTA → high RAM).",
        "  - Malformed/missing headers collapse into empty key; no warnings.",
        "[Explain] Edge cases",
        "  - Headers with spaces truncated at first whitespace.",
        "  - Non-FASTA lines are concatenated silently.",
        "[Explain] Next steps",
        "  - Add input validation + friendly errors.",
        "  - Add tests: empty file, multiline, odd headers. Consider streaming parser.",
    ];
    print_lines(&lines)
}

pub fn refactor(path: &str, target: Option<&str>, library: Option<&str>) -> Result<()> {
    let t = target.unwrap_or("python");
    let lib = library.unwrap_or("biopython");
    let lines = vec![
        &*format!("macbook:~/geneloop-demo $ geneloop refactor {} --target={} --library={}", path, t, lib),
        "Original (Perl, excerpt):",
        "------------------------------------------------------------",
        "open(my $fh, \"<\", $ARGV[0]) or die $!;",
        "my %seq; my $h = \"\";",
        "while (my $l = <$fh>) {",
        "  chomp $l;",
        "  if ($l =~ /^>(\\\\S+)/) { $h = $1; $seq{$h} = \"\"; }",
        "  else { $seq{$h} .= $l; }",
        "}",
        "for my $k (keys %seq) { print \"$k\\\\t\" . length($seq{$k}) . \"\\\\n\"; }",
        "Refactor (Python 3, Biopython):",
        "------------------------------------------------------------",
        "from __future__ import annotations",
        "from typing import Dict",
        "from Bio import SeqIO",
        "from pathlib import Path",
        "import sys",
        "",
        "def fasta_lengths(path: str | Path) -> Dict[str, int]:",
        "    records = SeqIO.parse(str(path), \"fasta\")",
        "    return {rec.id: len(rec.seq) for rec in records}",
        "",
        "if __name__ == \"__main__\":",
        "    if len(sys.argv) < 2:",
        "        sys.exit(\"usage: python fasta_lengths.py <file.fasta>\")",
        "    lengths = fasta_lengths(sys.argv[1])",
        "    for k, v in lengths.items():",
        "        print(f\"{k}\\\\t{v}\")",
        "Why safer: Biopython parser; typed; stream-based; easy tests.",
    ];
    print_lines(&lines)
}

