# Hybrid Structural-Generative Legal Recommendation Engine

Final-Year Project Report - Theoretical and Practical Analysis

## 1. Introduction

This project presents a legal recommendation system designed to map informal user queries to structured legal corrective actions while preserving factual integrity.

The system is built around one core rule:

- official legal content comes only from the validated dataset
- AI is used only to explain or communicate the result

The engine therefore:

- transforms natural-language queries into structured legal mapping
- ensures factual correctness through deterministic truth lookup
- prevents hallucination by constraining generation to explanation only

## 2. Theoretical Analysis

### 2.1 Compared Approaches

**Pure Generative RAG**

Flow:

`Query -> Retrieval -> Generated Answer`

Characteristics:

- flexible and conversational
- strong natural-language ability
- high hallucination risk in legal contexts

**Rule-Based Mapping**

Flow:

`Query -> Keyword Match -> Database`

Characteristics:

- fully deterministic
- easy to audit
- brittle when phrasing changes

**Hybrid Structural-Generative Model**

Flow:

`Query -> Normalization -> Retrieval -> NCid / truth group -> Deterministic lookup -> Explanation`

Characteristics:

- combines language flexibility with structured grounding
- keeps recommendations traceable to dataset records
- separates factual decision-making from language generation

### 2.2 Justification

The hybrid structural-generative approach was selected because neither pure generation nor pure rules are sufficient on their own for this use case.

Pure generative systems are attractive because they respond naturally to many forms of user input. However, in legal and compliance settings, a system that invents or distorts an official corrective action is unacceptable. Even a well-performing RAG system can still produce confident but incorrect wording if generation is allowed to decide the answer.

Rule-based systems sit at the opposite extreme. They are deterministic and easy to control, but they break down when users phrase the same issue in slightly different ways. A legal recommendation interface used by real users must tolerate paraphrase, extra wording, and imperfect wording.

The current design combines the strengths of both approaches while reducing their weaknesses:

- deterministic grounding ensures that every verified recommendation comes from the dataset
- semantic and lexical retrieval make the system robust to language variation
- controlled generation is used only after the official result is already selected
- explicit uncertainty modes prevent the system from pretending to know what it cannot verify

This design was chosen because it gives the best trade-off between:

- accuracy
- safety
- traceability
- usability

## 3. Practical Analysis

### 3.1 System Architecture

The system follows a hybrid layered architecture in which each layer has a narrow, explicit responsibility.

**1. Query Processing Layer**

- normalizes the user query
- removes wrapper phrasing and punctuation noise
- reduces wording variation before retrieval

**2. Truth Lookup Layer**

- loads the validated dataset
- indexes rows by `NCid`
- groups rows by normalized NC text
- checks whether an exact normalized match already exists
- resolves exact duplicates when NC and normalized Plan are the same
- returns ambiguity when the same NC maps to conflicting plans

**3. Retrieval Layer**

- uses BGE-M3 embeddings and Chroma when the vector index is available and in sync
- falls back safely to lexical retrieval when embeddings are unavailable or stale
- merges semantic and lexical candidates

**4. Reranking Layer**

- combines semantic score, fuzzy similarity, lexical overlap, containment bonus, and exact-match bonus
- applies a penalty to ambiguous candidate groups
- improves Top-1 selection quality

**5. Decision Layer**

- decides among `verified`, `ambiguous`, `advisory`, and `no_match`
- applies effective distance thresholds
- checks near-tie cases
- refuses unsafe certainty

**6. Truth Layer (Core Guarantee)**

- selects the official `NC`, `NCid`, and `Plan` from the dataset
- keeps recommendations dataset-grounded
- prevents the model from inventing legal actions

**7. Generation Layer**

- uses Vigogne through Ollama
- generates French explanations only after the factual result is already fixed
- also generates advisory wording when no official answer can be verified

### 3.2 Architecture Flow

Current runtime flow:

`User Query`
`-> Query Normalization`
`-> Exact Normalized Truth Lookup`
`-> Hybrid Retrieval (semantic + lexical fallback when needed)`
`-> Candidate Grouping by normalized NC`
`-> Hybrid Reranking`
`-> Decision Logic (verified / ambiguous / advisory / no_match)`
`-> Deterministic Truth Resolution`
`-> Prompt Construction`
`-> Vigogne Explanation`
`-> Final Structured Result`

Why this flow matters:

- exact lookup is tried before retrieval because it is the safest path
- retrieval is used only when exact lookup is insufficient
- reranking improves precision
- the decision layer protects against overconfident wrong answers
- generation happens last so it cannot alter the official result

### 3.3 Execution Pipeline

The main orchestration is implemented in `src/pipeline_runtime.py`.

Step-by-step execution:

**1. User query input**

- the user submits a natural-language description of a non-conformity

**2. Normalization**

- the query is cleaned and standardized
- wrapper phrases are removed
- the core legal issue is isolated

**3. Exact truth lookup**

- the system checks whether the normalized query directly matches a normalized NC in the dataset
- if one unique normalized plan exists, the result is immediately `verified`
- if several conflicting plans exist, the result becomes `ambiguous`

**4. Hybrid retrieval**

- if exact lookup is insufficient, the retriever searches candidates
- semantic retrieval uses embeddings when the Chroma index is valid
- lexical retrieval is always available as a safe fallback

**5. Candidate grouping**

- retrieved candidates are grouped by normalized NC text
- this reduces noise from repeated or near-duplicate rows

**6. Reranking**

- grouped candidates are scored using multiple signals
- the best candidate is selected only if confidence is sufficient

**7. Decision logic**

The pipeline chooses one of four result modes:

- `verified`: a strong unique official match was found
- `ambiguous`: official matches exist, but more than one conflicting plan remains
- `advisory`: the system cannot safely verify an official answer, but can provide a non-confirmed suggestion
- `no_match`: nothing close enough was found

**8. Truth resolution**

- if the selected group maps to one official plan family, the official row is returned
- if the selected group still contains conflicting plans, the result becomes `ambiguous`

**9. Generation**

- the model produces an explanation in French for `verified`
- the model produces a disclaimer-based suggestion for `advisory`
- ambiguity wording is template-based and grounded in the conflicting matches

### 3.4 File-Level Implementation

**`src/pipeline_runtime.py`**

- main orchestrator
- coordinates lookup, retrieval, reranking, decision logic, and generation
- enforces structured output modes instead of a single unchecked answer path

**`src/query_normalizer.py`**

- removes noisy wrapper phrasing from user questions
- improves exact lookup and retrieval quality

**`src/truth_lookup.py`**

- loads and standardizes the dataset
- normalizes NC and Plan text
- supports exact normalized lookup
- detects conflicting plan families
- supports curated overrides through `data/nc_resolution_overrides.csv`

**`src/retriever.py`**

- executes semantic retrieval through Chroma when available
- executes lexical retrieval through string and token similarity
- merges candidates into one ranked pool

**`src/reranker.py`**

- computes hybrid rerank scores
- improves ranking precision beyond raw vector search

**`src/generator.py`**

- calls Vigogne through Ollama
- generates explanation only after truth is fixed
- caches repeated explanations and advisory outputs

**`src/data_loader.py`**

- loads Excel data
- supports both old and updated schema column names
- canonicalizes data into `NCid`, `NC`, `Plan`

### 3.5 Key Technical Decisions

**1. Use of embeddings (BGE-M3)**

Chosen for:

- multilingual support
- strong semantic retrieval
- robustness to paraphrased legal wording

**2. Hybrid retrieval instead of embedding-only retrieval**

Chosen because:

- semantic search alone is not always sufficient
- lexical search improves resilience when wording is close
- lexical fallback keeps the system operational if embeddings or index sync fail

**3. Hybrid reranking**

Chosen because:

- the correct answer is often retrieved but not always ranked first
- combining multiple signals improves final candidate ordering

**4. Explicit multi-mode decision policy**

The runtime does not force every query into one answer.

Instead it returns:

- `verified`
- `ambiguous`
- `advisory`
- `no_match`

This is safer than a binary always-answer policy because it separates:

- trustworthy official answers
- conflicting official cases
- low-confidence suggestions
- missing matches

**5. Deterministic truth layer**

All verified official recommendations come from the dataset, not from the model.

This is the project's main factual guarantee.

## 4. Evaluation and Experimental Setup

### 4.1 Dataset

A benchmark dataset of 200 evaluation queries is used.

Each benchmark entry includes:

- user query
- expected `NCid`
- expected `NC`
- expected `Plan`

The production dataset is loaded automatically from the Excel files inside `data/`.

### 4.2 Evaluation Strategy

Two evaluation modes are used.

**1. Retrieval-Only Evaluation**

- generation disabled
- measures retrieval and ranking quality directly

Main metrics:

- `top1_ncid_accuracy`
- `top3_ncid_accuracy`
- `nc_text_accuracy`
- `plan_exact_match_rate`
- `verified_coverage`
- `verified_precision`
- `ambiguous_rate`
- latency

**2. Full Pipeline Evaluation**

- generation enabled
- measures the behavior of the complete runtime system

Main metrics:

- `pipeline_success_rate`
- `ncid_accuracy`
- `nc_text_accuracy`
- `plan_exact_match_rate`
- `verified_coverage`
- `verified_precision`
- `ambiguous_rate`
- `advisory_rate`
- `no_match_rate`
- average latency

## 5. Iterative Improvements

### 5.1 Earlier Baseline

The original baseline was simpler:

- semantic retrieval
- deterministic lookup by selected `NCid`
- explanation generation

Main limitations:

- weaker handling of wording variation
- weaker handling of duplicates and ambiguity
- excessive dependence on raw retrieval ordering

### 5.2 Query Normalization

Change:

- added query preprocessing to strip user wrapper phrasing

Result:

- improved exact lookup opportunities
- improved retrieval consistency

### 5.3 Hybrid Reranking

Change:

- added semantic + fuzzy + lexical reranking

Result:

- improved Top-1 selection
- reduced confusion between similar NCs

### 5.4 Exact Lookup Before Retrieval

Change:

- added normalized truth lookup before retrieval

Result:

- exact matches can be verified immediately
- direct ambiguity can be surfaced immediately
- safer than relying on retrieval for already-known cases

### 5.5 Ambiguity Handling

Change:

- introduced explicit `ambiguous` mode
- stopped flattening conflicting official plans into one answer

Result:

- safer system behavior
- more honest handling of dataset conflicts

### 5.6 Lexical Fallback and Index Guard

Change:

- retriever now supports lexical-only operation
- semantic retrieval is used only when the Chroma index matches the dataset

Result:

- more robust runtime behavior
- less dependence on infrastructure freshness

### 5.7 Curated Override Workflow

Change:

- added `data/nc_resolution_overrides.csv`
- added export script for conflict resolution

Result:

- ambiguous cases can be resolved operationally without hard-coding logic changes

## 6. Final Results

### 6.1 Retrieval-Only Evaluation

Current retrieval benchmark snapshot:

- Total queries: `200`
- Pipeline success rate: `100%`
- Top-1 NCid accuracy: `81%`
- Top-3 NCid accuracy: `96.5%`
- NC text accuracy: `84%`
- Plan exact match rate: `84.5%`
- Verified coverage: `85%`
- Ambiguous rate: `15%`
- Advisory rate: `0%`
- No-match rate: `0%`
- Verified precision: `95.29%`
- Average latency: `0.0014` seconds

### 6.2 Full Pipeline Evaluation

Current full pipeline benchmark snapshot:

- Total queries: `200`
- Pipeline success rate: `100%`
- NCid accuracy: `81%`
- NC text accuracy: `84%`
- Plan exact match rate: `84.5%`
- Verified coverage: `85%`
- Ambiguous rate: `15%`
- Advisory rate: `0%`
- No-match rate: `0%`
- Verified precision: `95.29%`
- Average latency: `5.1754` seconds
- Failed runs: `0`

## 7. Analysis

### 7.1 Retrieval vs Ranking

The evaluation shows that candidate recall is very strong:

- Top-3 NCid accuracy is `0.99`
- Top-1 NCid accuracy is `0.81`

This means the correct answer is usually being retrieved, but not always ranked first. The main technical challenge is therefore not simple recall; it is final candidate ordering and safe decision calibration.

### 7.2 Importance of Plan Match

The final business objective is not only matching the right label, but returning the right official corrective action.

For that reason, `plan_exact_match_rate` is one of the most meaningful metrics in the project.

### 7.3 Verified Precision

`verified_precision = 0.9529` means:

- when the system says a result is `verified`
- it is correct about `95.29%` of the time on `NCid`

This is a stronger trust metric than raw Top-1 accuracy alone.

### 7.4 Ambiguity as a Data Problem

The main remaining weakness is data ambiguity, not runtime instability.

Many difficult cases are not caused by a failure to retrieve candidates. They are caused by the dataset containing the same normalized NC with more than one legitimate plan family.

That is why the project now treats ambiguity as a first-class outcome instead of hiding it.

### 7.5 System Behavior

Current behavior is strong on:

- runtime stability
- factual grounding
- explicit uncertainty handling
- traceability of verified answers

The system does not hallucinate official plans because verified answers always come from the dataset.

## 8. Conclusion

This project demonstrates that a hybrid structural-generative approach can provide:

- strong factual safety
- useful language flexibility
- traceable official recommendations
- practical runtime performance

The core design decision is the separation between:

- truth selection
- language generation

Truth selection is handled by the structured dataset, lookup logic, retrieval, reranking, and decision policy.

Language generation is used only to explain a result that has already been selected safely.

That separation is what allows the system to remain usable without sacrificing legal reliability.

## 9. Current Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the vector index:

```bash
python main.py index
```

Active dataset used by the system:

`data/final_dedup_by_actionplan_recovered.xlsx`

Run a query:

```bash
python main.py query "absence d'autorisation administrative"
```
