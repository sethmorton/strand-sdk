Here you go — a single “concept doc” you can drop into Notion / Google Docs and tweak.

---

# Strand Concept Doc

*DNA Foundation Models as the “What Should We Test?” Layer*

---

## 1. How I Got Here (Short Journey)

* **Background:**
  I’m a CS / AI / RL person who got pulled into biology about a year ago. I’ve built multiple companies (including fintech), and I like building infrastructure, not one-off scripts.

* **First bio company:**
  “Cursor for DNA design” — interactive design platform + chatbot for constructs. This got me talking to *lots* of scientists and staring at the gap between cool models and what labs can actually use.

* **New obsession:** DNA foundation models (DNA FMs) & regulatory models
  Models like Evo2 / HyenaDNA / Enformer convinced me that we’re heading toward a world where **DNA FMs become the backbone of sequence understanding**, the way large language models are for text.

* **Shelby conversation:**
  Shelby (Compound) crystallized a few key realities:

  * Biotech has **tiny software budgets**, but **large outsourcing & compute budgets**.
  * DNA FMs are **underused**, especially in rare disease.
  * The biggest wins happen when you plug into **outsourced workflows** (tox, diagnostics, sequencing, functional studies), not when you ship “yet another tool”.

* **Strand so far:**
  I built **Strand SDK**: an open-source optimization engine for DNA. It can:

  * define environments + rewards
  * plug in DNA FMs and regulatory models
  * run CEM / GA / RL
  * output ranked candidates with full provenance.

I now want Strand to move from “cool SDK” → **something that owns a specific, painful decision in the bio stack.**

---

## 2. The Idea Itself

### 2.1 One-Liner

**Strand builds the “what should we actually test?” layer for genomics — using DNA foundation models and virtual-cell models to design higher-yield experiments, starting with regulatory panels for rare disease and functional genomics.**

---

### 2.2 The Problem

Modern genomics teams are good at **sequencing**:

* WES/WGS, panels, whole genomes → thousands of variants per patient.
* GWAS & eQTL → broad regions of noncoding DNA implicated in disease.
* High-throughput assays (MPRA, CRISPR tiling) can test thousands of sequences at once.

But they can’t test everything:

* MPRAs and CRISPR screens are **expensive**, **labor-intensive**, and **limited in capacity**.
* Labs often have **tens of thousands** of possible variants/tiles but budget for a few thousand.

Today, the actual selection is done with:

* ad-hoc notebooks & scripts
* conservation + a few scores + human intuition
* scattered knowledge in people’s heads and emails.

The core decision that’s under-optimized:

> “Given all the variants/regions/constructs we *could* test and a fixed assay budget, **which ones should we put in this experiment?**”

---

### 2.3 Why DNA FMs & Virtual-Cell Models Matter Here

We now have:

* **DNA FMs**: big models trained on genomes that encode rich priors about what real genomic DNA looks like.
* **Virtual-cell models** (e.g. Enformer): tools that predict regulatory activity / expression from sequence.

These models are:

* perfectly set up to tell us **which sequences are likely to matter** in a given locus / tissue,
* but they’re rarely wired into the **actual experimental design step**.

Everyone uses them to score things. Very few use them to **choose** what to test under real constraints.

That’s Strand’s wedge.

---

### 2.4 What Strand Does (Conceptually)

Strand’s core job:

> Turn “here are all the variants/regions we might care about”
> into
> “here is the assay-ready panel/library we should actually synthesize and test”
> **under a budget**, using DNA FMs + virtual cell models + deterministic features.

For v1, the flagship campaign is:

> **DNA-FM Virtual Cell Regulatory Panel Design**
> A campaign that picks MPRA/CRISPR tiling constructs for regulatory variants so that, at fixed panel size, you capture **more truly functional variants** than baseline strategies.

#### Inputs (for the campaign)

* Candidate variants / tiles (from rare-disease cohorts, GWAS regions, etc.)
* Genomic coordinates + sequences
* Assay type (MPRA promoter/enhancer, CRISPRi/a tiling)
* Budget (max number of constructs)
* Relevant tissues / context where possible

#### Engine (Strand + models)

* Virtual cell model (e.g. Enformer) for:

  * ∆ predicted regulatory signal between ref/alt
  * context-specific regulatory features
* DNA FM(s) for:

  * sequence priors
  * plausibility / novelty in latent space
* Deterministic features:

  * motif hits, motif gain/loss
  * conservation
  * basic sequence QC
* Strand SDK:

  * wraps these as **RewardBlocks**
  * runs scoring / search strategies
  * selects a panel that maximizes expected “functional value” under the assay budget.

#### Outputs

* **Panel manifest**:

  * each construct’s sequence
  * coordinates / variant IDs
  * model-based annotations
  * priority ranking

* **Summary report** with:

  * why these constructs
  * coverage of loci/variants
  * how models contributed

* **Provenance package**:

  * model versions, configs, parameters

---

### 2.5 Why This Is Useful (Not Just a Gimmick)

* For labs already running MPRAs/CRISPR tiling, this decision is **real and expensive**.
* Better panels mean:

  * more hits per experiment
  * fewer constructs wasted
  * more information per dollar.

Instead of shipping “yet another score”, Strand:

* **owns the decision** of what gets tested next,
* and uses DNA FMs / virtual cells as the brain behind that decision.

The v1 campaign is designed so you can **prove this offline** on public MPRA datasets:

* treat the full tested library as the universe
* choose K constructs as if pre-experiment
* use the MPRA readouts as ground truth
* show that Strand chooses panels with more functional sequences than:

  * random
  * simple heuristics (conservation)
  * single-model ranking.

That’s your first “Tempus/Cytely-style” plot.

---

## 3. Exploration of Other Ideas (and Why This One)

### 3.1 Variant triage score

**Idea:** take WES/WGS variants, stack Enformer + DNA FM + deterministic features, and produce a better ranking for rare-disease variant interpretation.

**Why it was tempting:**

* Directly aligns with Shelby’s excitement about DNA FMs + rare disease.
* Very natural for a CS/ML person: “Let’s build a better classifier.”

**Why we backed away:**

* Clinical pipelines already have **many predictors**; a new score looks like “tool #29”.
* ACMG-style frameworks treat in silico evidence as **supporting**, not decisive.
* You become “another column” instead of “a step in the workflow”.
* Hard to map into **outsourcing/compute** budgets; it still feels like SaaS tooling.

In short: technically cool, commercially and workflow-wise mid.

---

### 3.2 Generic RL post-training / model-postprocessing platform

**Idea:** generic “RL and post-training for DNA FMs” — let anyone plug in their model and define arbitrary reward functions for optimization.

**Why it fits you:**

* Perfect match to your RL / infra strengths.
* Elegant abstraction (environments, strategies, reward blocks).

**Why it’s not enough alone:**

* It sells **capability**, not a **job**.
* Bio orgs don’t wake up thinking “I need a model post-training platform,” they think “I need to decide which sequences/variants to test or ship.”
* Hard to tie to a specific line item or hair-on-fire problem on its own.

This became the **Strand SDK**, which is great — but you need a focused campaign on top.

---

### 3.3 Big, later-stage markets: tox, gene therapy, DELs

We explored:

* **Toxicity / safety** for oligo/gene therapies:

  * huge outsourced budgets
  * strong fit with “outsourcing + compute”
  * but messy multi-modal biology → high risk of overselling early.

* **Gene therapy / AAV library design**:

  * big spend, AI players already emerging
  * DNA FMs somewhat relevant but domain is very specific
  * more competitive, and more protein/structure-driven.

* **DEL / small-molecule-land**:

  * misaligned with DNA FMs (focus is small molecules, not DNA itself).

**Conclusion:**

These are great **Phase 2/3** directions once:

* the engine is proven,
* you’ve built trust,
* and maybe you have wet-lab partners.

As a **Phase 1 proving ground**, regulatory panel design:

* is small but real,
* is cleanly aligned with DNA FMs + virtual-cell models,
* and can be validated fully offline on public datasets.

So we settled on:

> “Start with regulatory experiment design (MPRA/CRISPR), prove the concept, then expand to other assays and safety.”

---

## 4. Roadmap for Strand

### Phase 1 — Proof on Public Data (0–3 months)

**Goal:** Show Strand’s DNA-FM + virtual-cell campaign can design better regulatory panels on public MPRA datasets.

* Choose 1–2 MPRA datasets (via MPRAVarDB etc.).
* Build a clean feature table:

  * sequences, MPRA readouts, functional labels, Enformer features, DNA FM features, motifs, conservation.
* Define strategies:

  * random
  * simple heuristic (conservation)
  * single-model (Enformer-only)
  * **Strand campaign** (multi-block reward).
* Evaluate:

  * hits per panel at fixed sizes
  * recall of functional hits
  * enrichment vs random.
* Ship:

  * example folder / repo
  * case study page on strand.tools
  * simple plots that show clear uplift.

### Phase 1.5 — Design-Only Pilots with Real Teams (3–9 months)

**Goal:** Run a few real campaigns for teams that already have MPRAs/CRISPR tiling.

* Package **Strand Regulatory Panel Design** as a service:

  * standard inputs & outputs
  * estimated turnaround time.
* Offer pilot projects:

  * design panels for their next regulatory screen
  * get post-hoc feedback once they run it.
* Use results to:

  * refine reward functions
  * gather anecdotal “we found these hits because of Strand” stories
  * tune UX & data formats.

### Phase 2 — Partnered Experiment Design (9–18 months)

**Goal:** Become a true “variant-to-function” outsourcing step via partners.

* Partner with 1–2 wet labs / functional genomics CROs running MPRAs/CRISPR.
* Offer end-to-end:

  * Strand designs panel → partner runs screen → combined report to customer.
* Now you sit clearly in the **outsourcing** bucket.
* Every project enriches your dataset:

  * variants + design + ground truth, perfect for DNA FM / virtual-cell training and calibration.

### Phase 3 — Additional Campaigns & Expansion (18+ months)

Build additional “Strand campaigns” on the same engine, for example:

* safety / immunogenicity filtering for NA therapeutics
* toxicity prefiltering for certain libraries
* promoter/enhancer design for expression in specific tissues
* off-target / regulatory-risk panels for gene therapies

At this stage, Strand looks like:

> A platform of **experiment design campaigns**, all powered by DNA FMs + virtual-cell models + Strand SDK.

---

## 5. Insights from Tempus, Cytely, and Shelby

### 5.1 From Tempus

* They make **most of their money from diagnostics**, not “AI features.”
* AI runs inside the pipes; the product is a **standardized test** with billing codes, reports, and EMR integration.
* Lessons for Strand:

  * Don’t sell “AI” → sell a **clear, repeated workflow**.
  * Standardize inputs/outputs like a test or design SKU.
  * Long-term, aim for a **data flywheel** where every experiment improves your models.

### 5.2 From Cytely

* Cytely doesn’t say “we do AI for images”; they say:

  * “We turn your microscope into an intelligent, high-throughput system.”

* They own a **specific lab workflow** (microscopy) and promise hard numbers (throughput, speed).

* Lessons for Strand:

  * Anchor to **experiment design** as a named workflow, not vague “optimization”.
  * Show **concrete uplifts**: more functional hits per construct, fewer constructs for same hit count.
  * Be opinionated: “this is how experiment design is done if you take DNA FMs seriously.”

### 5.3 From Shelby

Key lines that shape this whole thing:

* Biotech has **tiny software budgets**, but **big outsourcing and compute budgets**.
* DNA FMs are **underrated** and underused in rare disease + sequence interpretation.
* Rare disease + foundational DNA models are a huge opportunity if you pick a **hair-on-fire** job.
* Wet-lab validation, or pairing with wet-lab people, is how you win trust in a skeptical field.

Implications:

* Strand should present as:

  * **“We do this experiment-design job for you”**, not “here’s a tool.”
  * A **compute-heavy backend** they don’t have to maintain.
  * Something that could plug into **outsourcing** flows, especially once you have lab partners.

### 5.4 DNA FMs as the Center of Gravity

The emerging worldview:

* DNA FMs will become the **default representation** for genomic sequence.
* Virtual-cell models will become the **default oracles** for regulatory behavior.
* What’s missing is the **optimization + provenance layer** that:

  * sits between these models and wet lab
  * understands budgets, constraints, and objectives
  * and answers:

    > “Given everything we know and all the models we have, what should we actually test next?”

That’s the role Strand is trying to claim.

---

If you want, next step I can help you turn this into:

* a short internal “conviction memo” you share with close advisors, or
* a public-facing “Why I’m building Strand” blog post that uses a subset of this in a more narrative tone.
