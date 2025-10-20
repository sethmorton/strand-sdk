// Run immediately (script at end of body)
(function(){
  const themeToggle=document.getElementById('themeToggle');
  (function initTheme(){
    const saved = localStorage.getItem('gl-theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = saved || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
    themeToggle.textContent = theme === 'dark' ? 'Light' : 'Dark';
  })();
  themeToggle.addEventListener('click', ()=>{
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('gl-theme', next);
    themeToggle.textContent = next === 'dark' ? 'Light' : 'Dark';
  });

  const tabs = Array.from(document.querySelectorAll('.tab'));
  const panels = {
    debug: document.getElementById('panel-debug'),
    gen: document.getElementById('panel-gen'),
    explain: document.getElementById('panel-explain'),
    refactor: document.getElementById('panel-refactor'),
  };

  function showThinking(panel){
    const t = panel.querySelector('.thinking');
    if(!t) return Promise.resolve();
    const icon = t.querySelector('.icon');
    const word = t.querySelector('.word');
    const words = []; // no rotating words; keep it minimal
    const icons = ['•','✢','✳','*','✻','✽'];
    let wi=0, ii=0;
    t.style.display='flex';
    const ivI = setInterval(()=>{ icon.textContent = icons[ii++%icons.length]; }, 300);
    const ivW = setInterval(()=>{ word.textContent = ''; }, 600);
    // Slightly slower: ~2.4s – 4.2s
    const duration = Math.random()*1800 + 2400;
    return new Promise(resolve=>{ setTimeout(()=>{ clearInterval(ivI); clearInterval(ivW); t.style.display='none'; resolve(); }, duration); });
  }

  async function typeOut(panel, lines){
    const out = panel.querySelector('.output');
    out.innerHTML='';
    const sleep=(ms)=> new Promise(r=>setTimeout(r,ms));
    for(const item of lines){
      const ln = typeof item === 'string' ? item : (item.text || '');
      const d=document.createElement('div'); d.className='line'; d.textContent=ln; out.appendChild(d);
      panel.scrollTop = panel.scrollHeight;
      // Slightly slower per line for readability
      await sleep(160);
      if(typeof item === 'object' && item.pauseMs){ await sleep(item.pauseMs); }
    }
    // No sticky closing prompt in showcase to avoid overlap/duplication
  }

  const scripts = {
    // Lead line (shown before thinking), then the rest
    debugLead: [
      'macbook:~/geneloop-demo $ geneloop how can I debug this file? sample.bam'
    ],
    debug: [
      '[GeneLoop] Parsed intent → geneloop debug sample.bam',
      '[GeneLoop] Scanning filesystem (cwd, neighbors)… ok',
      '[GeneLoop] Detecting file type…      ok (BAM)',
      '[GeneLoop] Quick sanity checks…      ok (sorted: coordinate)',
      '[GeneLoop] Index presence…           missing (.bai not found)',
      '[GeneLoop] Read summary (fast scan):',
      '  - total reads: ~42,317,894',
      '  - mapped: 97.6%   | properly paired: 95.8%',
      '  - duplicates: 11.9% (flag-based)',
      '  - multimappers: 6.3%',
      '  - median insert size: 286 bp',
      '  - GC content (reads): 42% | reference: 41%',
      '[Model] Thinking…',
      '[Model] Hypothesis: BAM is sorted; missing sidecar index (.bai).',
      '[Model] Evidence: SO:coordinate in header; .bai absent; typical idxstats/IGV failure.',
      '[Model] Plan: propose samtools index; re-check; warn on multimappers/GC.',
      '[Issue 1] Missing BAM index',
      '  Fix: samtools index -@ 8 sample.bam',
      '[Issue 2] Multimapper noise (6.3%) — consider MAPQ >=10 or STAR',
      '[Note] GC bias (coarse) — no strong deviation observed',
      '[GeneLoop] Suggested next command:',
      '  samtools index -@ 8 sample.bam && samtools idxstats sample.bam | head',
      { text: '[Decision] Auto-generate fix command? [y/N]:', pauseMs: 2200 },
      { text: 'y', pauseMs: 600 },
      'Running: samtools index -@ 8 sample.bam',
      '  writing index… done',
      '[Index] Done: sample.bam.bai (2.1s)',
      'Re-checking…',
      '[GeneLoop] Loading index… ok',
      '[GeneLoop] Index presence…           ok',
      '[GeneLoop] Ready for IGV / variant calling.',
      'Summary',
      '  Root cause: Missing .bai after sort',
      '  Fix applied: samtools index -@ 8 sample.bam',
      '  Risk notes: multimappers ~6.3%; consider MAPQ filter (>=10)'
    ],
    genLead: [
      'macbook:~/geneloop-demo $ geneloop build me a simple rna-seq pipeline'
    ],
    gen: [
      '[GeneLoop] Parsed intent → geneloop gen "FASTQ to BAM: paired-end RNA-seq; hg38"',
      'macbook:~/geneloop-demo $ geneloop gen "FASTQ to BAM: paired-end RNA-seq; hg38"',
      '[GeneLoop] Parsing intent… ok',
      '[Model] Reasoning: paired-end RNA‑seq; hg38; prefer bwa-mem2 for demo speed.',
      '[Model] Plan: emit minimal idempotent slice + environment pins.',
      '[GeneLoop] Resolving toolchain (bwa-mem2, samtools)… ok',
      '[GeneLoop] Emitting idempotent Snakemake slice (FASTQ ➜ BAM)',
      '# Snakefile',
      'rule all:',
      '    input: expand("results/bam/{sample}.bam.bai", sample=lambda: config[\'samples\'])',
      '',
      'rule align_and_sort:',
      '    input:',
      '        r1="data/fastq/{sample}_R1.fastq.gz",',
      '        r2="data/fastq/{sample}_R2.fastq.gz",',
      '        ref="refs/hg38.fa"',
      '    output:',
      '        bam=temp("results/bam/{sample}.bam"),',
      '        bai="results/bam/{sample}.bam.bai"',
      '    params:',
      '        threads=8,',
      '        bwa_opts="-R \'@RG\\tID:{wildcards.sample}\\tSM:{wildcards.sample}\\tPL:ILLUMINA\'"',
      '    log: "logs/{sample}.align.log"',
      '    conda: "envs/aln.yaml"',
      '    shell:',
      '        r"""',
      '        set -eo pipefail',
      '        if [ ! -s {output.bam} ]; then',
      '          bwa-mem2 mem -t {params.threads} {params.bwa_opts} {input.ref} {input.r1} {input.r2} 2>> {log} | \\',
      '          samtools view -@ {params.threads} -bS - | \\',
      '          samtools sort -@ {params.threads} -o {output.bam} - 2>> {log}',
      '        fi',
      '        samtools index -@ {params.threads} {output.bam} 2>> {log}',
      '        """',
      '',
      '# envs/aln.yaml',
      'name: biocode-aln',
      'channels: [conda-forge, bioconda]',
      'dependencies:',
      '  - bwa-mem2=2.*',
      '  - samtools=1.19.*',
      '',
      '# config.yaml',
      'samples:',
      '  - sampleA',
      '  - sampleB',
      '[GeneLoop] Files written: Snakefile, envs/aln.yaml, config.yaml (virtual).',
      'Notes: Replace BWA-MEM2 with STAR for spliced RNA‑seq if preferred.',
      'Idempotent: no re-run if outputs exist; index enforced.'
    ],
    explainLead: [
      'macbook:~/geneloop-demo $ geneloop what does this do? legacy/fasta_parser.pl'
    ],
    explain: [
      '[GeneLoop] Parsed intent → geneloop explain legacy/fasta_parser.pl',
      'macbook:~/geneloop-demo $ geneloop explain legacy/fasta_parser.pl',
      '[GeneLoop] Reading file: legacy/fasta_parser.pl…',
      '[Explain] Summary',
      '  - Purpose: parse FASTA records and print id\tlength TSV.',
      '  - Flow: detect headers (">id"), accumulate sequence lines per id, output lengths.',
      '[Explain] Assumptions',
      '  - Header regex: /^(\\S+)/ captures ID without spaces.',
      '  - Multiline sequences supported; blank lines ignored.',
      '[Explain] Pitfalls',
      '  - Memory growth: stores entire sequences in a hash (large FASTA → high RAM).',
      '  - Malformed/missing headers collapse into empty key; no warnings.',
      '[Explain] Edge cases',
      '  - Headers with spaces truncated at first whitespace.',
      '  - Non-FASTA lines are concatenated silently.',
      '[Explain] Next steps',
      '  - Add input validation + friendly errors.',
      '  - Add tests: empty file, multiline, odd headers. Consider streaming parser.'
    ],
    refactorLead: [
      'macbook:~/geneloop-demo $ geneloop can you refactor this to python? legacy/fasta_parser.pl'
    ],
    refactor: [
      '[GeneLoop] Parsed intent → geneloop refactor legacy/fasta_parser.pl --target=python --library=biopython',
      'macbook:~/geneloop-demo $ geneloop refactor legacy/fasta_parser.pl --target=python --library=biopython',
      'Original (Perl, excerpt):',
      '------------------------------------------------------------',
      'open(my $fh, "<", $ARGV[0]) or die $!;',
      'my %seq; my $h = "";',
      'while (my $l = <$fh>) {',
      '  chomp $l;',
      '  if ($l =~ /^>(\\S+)/) { $h = $1; $seq{$h} = ""; }',
      '  else { $seq{$h} .= $l; }',
      '}',
      'for my $k (keys %seq) { print "$k\\t" . length($seq{$k}) . "\\n"; }',
      'Refactor (Python 3, Biopython):',
      '------------------------------------------------------------',
      'from __future__ import annotations',
      'from typing import Dict',
      'from Bio import SeqIO',
      'from pathlib import Path',
      'import sys',
      '',
      'def fasta_lengths(path: str | Path) -> Dict[str, int]:',
      '    records = SeqIO.parse(str(path), "fasta")',
      '    return {rec.id: len(rec.seq) for rec in records}',
      '',
      'if __name__ == "__main__":',
      '    if len(sys.argv) < 2:',
      '        sys.exit("usage: python fasta_lengths.py <file.fasta>")',
      '    lengths = fasta_lengths(sys.argv[1])',
      '    for k, v in lengths.items():',
      '        print(f"{k}\\t{v}")',
      'Why safer: Biopython parser; typed; stream-based; easy tests.'
    ]
  };

  const played = { debug:false, gen:false, explain:false, refactor:false };
  async function activate(name){
    tabs.forEach(btn=> btn.setAttribute('aria-selected', btn.dataset.panel===name ? 'true':'false'));
    Object.entries(panels).forEach(([k,el])=>{ el.hidden = k!==name; });
    const panel = panels[name];
    panel.scrollTop=0;
    if(!played[name]){
      const leadKey = name + 'Lead';
      if(scripts[leadKey]){ await typeOut(panel, scripts[leadKey]); }
      await showThinking(panel);
      await typeOut(panel, scripts[name]);
      played[name]=true;
    }
  }

  tabs.forEach(btn=> btn.addEventListener('click', ()=> activate(btn.dataset.panel)));
  activate('debug');
})();
