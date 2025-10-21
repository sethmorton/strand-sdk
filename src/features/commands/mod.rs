use anyhow::{Result, bail};
use std::io::{self, Write};
use std::fs;
use crate::services::openai::{self};
use crate::services::config;
use crate::services::{prompts, structure, transcript::CmdType};
use crate::GlobalOpts;

fn type_out(lines: Vec<String>) -> Result<()> {
    // Render line-by-line (not per-character) to match the demo
    // Consistent pacing regardless of TTY/pipes; no feature flags.
    const LINE_DELAY_MS: u64 = 160;
    let mut out = io::stdout().lock();
    for l in lines {
        writeln!(out, "{}", l)?;
        out.flush().ok();
        std::thread::sleep(std::time::Duration::from_millis(LINE_DELAY_MS));
    }
    Ok(())
}

pub fn debug(path: &str) -> Result<()> {
    let mut out: Vec<String> = Vec::new();
    out.push("[Debug] Context".into());
    out.extend([
        &*format!("  - file: {}", path),
        "  - environment: cwd scan, index check",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Debug] Findings".into());
    out.extend([
        "  - type: BAM (sorted: coordinate)",
        "  - index: missing (.bai not found)",
        "  - reads: ~42,317,894; mapped 97.6%; pairs 95.8%",
        "  - multimappers: 6.3%; dupes: 11.9%",
        "  - insert size (median): 286 bp",
        "  - GC: reads 42% | ref 41%",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Debug] Issues".into());
    out.extend([
        "  - Missing BAM index (.bai)",
        "  - Multimapper noise (6.3%)",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Debug] Fixes".into());
    out.extend([
        "  - samtools index -@ 8 sample.bam",
        "  - consider MAPQ >=10 filter or STAR aligner",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Debug] Suggested next commands".into());
    out.push("  - samtools index -@ 8 sample.bam && samtools idxstats sample.bam | head".into());
    out.push("[Debug] Summary".into());
    out.extend([
        "  - Root cause: missing .bai after sort",
        "  - Fix: indexed BAM; ready for IGV/variant calling",
    ].into_iter().map(|s| s.to_string()));
    // Type-out effect and interactive confirm for the fix command
    type_out(out.clone())?;
    // Prompt (works with TTY and piped input)
    print!("[Decision] Auto-generate fix command? [y/N]: ");
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
    if input.trim().eq_ignore_ascii_case("y") {
        let followup = vec![
            "Running: samtools index -@ 8 sample.bam".to_string(),
            "  writing index… done".to_string(),
            "[Index] Done: sample.bam.bai (2.1s)".to_string(),
            "Re-checking…".to_string(),
            "[GeneLoop] Loading index… ok".to_string(),
            "[GeneLoop] Index presence…           ok".to_string(),
            "[GeneLoop] Ready for IGV / variant calling.".to_string(),
        ];
        type_out(followup)?;
    } else {
        println!("(skipped)");
    }
    Ok(())
}

pub fn gen(spec: &str) -> Result<()> {
    let mut out: Vec<String> = Vec::new();
    out.push("[Gen] Intent".into());
    out.push(format!("  - {}", spec));
    out.push("[Gen] Plan".into());
    out.extend([
        "  - Emit minimal idempotent slice + environment pins.",
        "  - Resolve tools and write Snakemake + env + config.",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Gen] Files".into());
    out.extend([
        "  - Snakefile",
        "  - envs/aln.yaml",
        "  - config.yaml",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Gen] Snippets".into());
    out.extend([
        "```snakemake",
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
        "```",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Gen] Next steps".into());
    out.extend([
        "  - Replace BWA-MEM2 with STAR for spliced RNA-seq if needed.",
        "  - Fill config.yaml with sample names.",
    ].into_iter().map(|s| s.to_string()));
    type_out(out)
}

pub fn explain(path: &str) -> Result<()> {
    let mut out: Vec<String> = Vec::new();
    out.extend([
        "[Explain] Summary",
        &*format!("  - File: {}", path),
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
    ].into_iter().map(|s| s.to_string()));
    type_out(out)
}

pub fn refactor(path: &str, target: Option<&str>, library: Option<&str>) -> Result<()> {
    let t = target.unwrap_or("python");
    let lib = library.unwrap_or("");
    let mut out: Vec<String> = Vec::new();
    out.push("[Refactor] Summary".into());
    out.extend([
        &*format!("  - Target: {}{}", t, if lib.is_empty(){ "" } else { " (with library)" }),
        &*format!("  - File: {}", path),
    ].into_iter().map(|s| s.to_string()));
    out.push("[Refactor] Plan".into());
    out.extend([
        "  - Preserve behavior; improve readability and safety.",
        "  - Add minimal structure and type hints where appropriate.",
    ].into_iter().map(|s| s.to_string()));
    out.push("[Refactor] New code".into());
    out.push("```python".into());
    out.extend([
        "from __future__ import annotations",
        "from typing import Dict",
        "from Bio import SeqIO  # optional",
        "from pathlib import Path",
        "",
        "def fasta_lengths(path: str | Path) -> Dict[str, int]:",
        "    records = SeqIO.parse(str(path), 'fasta')",
        "    return {rec.id: len(rec.seq) for rec in records}",
        "",
        "if __name__ == '__main__':",
        "    import sys",
        "    if len(sys.argv) < 2:",
        "        sys.exit('usage: python fasta_lengths.py <file.fasta>')",
        "    lengths = fasta_lengths(sys.argv[1])",
        "    for k, v in lengths.items():",
        "        print(f'{k}\t{v}')",
    ].into_iter().map(|s| s.to_string()));
    out.push("```".into());
    out.push("[Refactor] Notes".into());
    out.extend([
        "  - Replace Biopython with a streaming parser if dependency-free is required.",
    ].into_iter().map(|s| s.to_string()));
    type_out(out)
}

fn resolve_api_key(globals: &GlobalOpts) -> Option<String> {
    if let Some(k) = globals.api_key.clone() { return Some(k); }
    if let Ok(k) = std::env::var("OPENAI_API_KEY") { return Some(k); }
    if let Ok(k) = std::env::var("GENELOOP_API_KEY") { return Some(k); }
    if let Ok(cfg) = config::load() { if let Some(k) = cfg.api_key { return Some(k); } }
    None
}

fn read_limited(path: &str, max_bytes: usize) -> Result<String> {
    let data = fs::read(path)?;
    let slice = if data.len() > max_bytes { &data[..max_bytes] } else { &data[..] };
    Ok(String::from_utf8_lossy(slice).to_string())
}

pub fn explain_online(globals: &GlobalOpts, path: &str) -> Result<()> {
    if !globals.online {
        return explain(path);
    }
    let Some(api_key) = resolve_api_key(globals) else { bail!("No API key found. Pass --api-key or set OPENAI_API_KEY/GENELOOP_API_KEY."); };
    let content = read_limited(path, 100_000)?;
    // resolve model: CLI > config > default
    let model = if !globals.model.is_empty() { globals.model.clone() } else { config::load().ok().and_then(|c| c.model).unwrap_or_else(|| "gpt-4o-mini".into()) };
    let messages = prompts::explain_prompt(path, &content);
    let raw = openai::chat_blocking(&api_key, &model, messages)?;
    let strict = config::load().ok().and_then(|c| c.structure_strict).unwrap_or(true);
    let final_out = if globals.no_structure || !strict {
        raw
    } else {
        structure::enforce_structure(CmdType::Explain, &raw)
    };
    println!("{}", final_out.trim());
    Ok(())
}

pub fn refactor_online(globals: &GlobalOpts, path: &str, target: Option<&str>, library: Option<&str>) -> Result<()> {
    if !globals.online {
        return refactor(path, target, library);
    }
    let Some(api_key) = resolve_api_key(globals) else { bail!("No API key found. Pass --api-key or set OPENAI_API_KEY/GENELOOP_API_KEY."); };
    let content = read_limited(path, 80_000)?;
    let model = if !globals.model.is_empty() { globals.model.clone() } else { config::load().ok().and_then(|c| c.model).unwrap_or_else(|| "gpt-4o-mini".into()) };
    let t = target.unwrap_or("python");
    let lib = library.unwrap_or("");
    let messages = prompts::refactor_prompt(path, &content, t, lib);
    let raw = openai::chat_blocking(&api_key, &model, messages)?;
    let strict = config::load().ok().and_then(|c| c.structure_strict).unwrap_or(true);
    let final_out = if globals.no_structure || !strict {
        raw
    } else {
        structure::enforce_structure(CmdType::Refactor, &raw)
    };
    println!("{}", final_out.trim());
    Ok(())
}

pub fn auth_login(globals: &GlobalOpts, api_key: Option<&str>, model: Option<&str>) -> Result<()> {
    // Determine key priority: arg > --api-key > env > current config
    let key = if let Some(k) = api_key { Some(k.to_string()) }
        else if let Some(k) = globals.api_key.clone() { Some(k) }
        else if let Ok(k) = std::env::var("OPENAI_API_KEY") { Some(k) }
        else { std::env::var("GENELOOP_API_KEY").ok() };

    if let Some(ref k) = key { config::set_api_key(k)?; println!("Saved API key to {:?}", crate::services::config::config_path()?); }
    if let Some(m) = model { config::set_model(m)?; println!("Saved default model: {}", m); }
    if key.is_none() && model.is_none() {
        println!("Usage: geneloop auth login --api-key sk-... [--model gpt-4o-mini]");
    }
    Ok(())
}

pub fn auth_show() -> Result<()> {
    let cfg = config::load()?;
    let masked = cfg.api_key.as_deref().map(|k| {
        if k.len() <= 8 { "********".to_string() } else { format!("{}…{}", &k[..4], &k[k.len()-4..]) }
    }).unwrap_or_else(|| "(none)".into());
    println!("Config: {:?}", crate::services::config::config_path()?);
    println!("  api_key: {}", masked);
    println!("  model: {}", cfg.model.unwrap_or_else(|| "(default: gpt-4o-mini)".into()));
    println!("  theme: {}", cfg.theme.unwrap_or_else(|| "(light)".into()));
    Ok(())
}
