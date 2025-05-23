# Longform Creative Writing Benchmark

A comprehensive benchmark for evaluating language models' abilities in creative writing, planning, and narrative construction. This benchmark tests models on their capacity to brainstorm, plan, revise, and write complete short stories/novellas from minimal prompts.

This codebase can be used to reproduce results on: https://eqbench.com/creative_writing_longform.html

## Overview

The benchmark evaluates several key abilities:
- **Brainstorming & Planning**: Creating a coherent story plan from a minimal prompt
- **Critical Reflection**: Reviewing and revising the initial plan
- **Character Development**: Creating detailed character profiles
- **Long-form Writing**: Producing a complete novella across 8 chapters (~1000 words each)
- **Narrative Consistency**: Maintaining plot coherence and character consistency throughout

Models on the EQ-Bench leaderboard are evaluated with Claude Sonnet 3.7 as a judge, although you can use whichever judge you wish. The judge scores outputs across multiple criteria including creativity, coherence, character development, and prose quality.

---

## Quick start

```bash
# 1. Install deps (prefer venv)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Copy environment template and fill in keys
cp .env.example .env
$EDITOR .env            # set TEST_API_KEY & JUDGE_API_KEY, or OPENAI_API_KEY

# 3. Run one prompt, one iteration, four OS threads
python3 longform_writing_bench.py \
    --test-model  "google/gemini-2.0-flash-001" \
    --judge-model "anthropic/claude-3.7-sonnet" \
    --runs-file   "results/longform_bench_runs.json" \
    --run-id      "demo" \
    --threads     12 \
    --iterations  1
````

> **Tip** The sample above assumes OpenRouter endpoints (identical payload shape to OpenAI).
> If you point `TEST_API_URL` or `JUDGE_API_URL` elsewhere, adjust headers in `utils/api.py` if needed.

---

## Environment variables (`.env`)

| key                               | default   | purpose                                     |
| --------------------------------- | --------- | ------------------------------------------- |
| `TEST_API_URL` / `TEST_API_KEY`   | –         | endpoint & key for the model under test     |
| `JUDGE_API_URL` / `JUDGE_API_KEY` | –         | endpoint & key for the judge model          |
| `MAX_RETRIES`                     | `5`       | per-request retry limit                     |
| `RETRY_DELAY`                     | `5` (s)   | base delay between retries (doubles on 429) |
| `REQUEST_TIMEOUT`                 | `300` (s) | hard timeout for any HTTP request           |
| `LOG_VERBOSITY`                   | `INFO`    | fallback log level (CLI `--verbosity` wins) |

Only these six are required; everything else has sane defaults.

---

## What happens during a run

1. **Initialization**

   * generates a unique `run_key` `<run-id>_<sanitized-model-name>`
   * writes skeleton entry in `results/longform_bench_runs.json`

2. **Generation (13 steps)**

   * Prompts 1-5: plan and character profiles
   * Prompts 6-13: eight \~1000-word chapters
   * Uses `temperature=0.7`, `min_p=0.1` by default
   * Saves after every `--save-interval` steps (default 1)

3. **Chapter judging**

   * For each chapter, Claude (or your judge) responds with a rubric and scores (0-20 scale)

4. **Final judging**

   * Judge sees the full story once (configurable via `NUM_FINAL_JUDGMENTS`) and outputs another score block.

5. **Scoring & bootstrap**

   * Weights chapter scores equally; the final piece can have its own weight (`FINAL_SCORE_WEIGHT`).
   * Calculates mean ±95 % CI from 500 bootstrap resamples.
   * Appends the result under `runs.<run_key>.results.benchmark_results`.

All I/O uses atomic writes with per-file locks (`utils/file_io.py`), so parallel threads and crashes won’t corrupt logs.

---

## Resuming / re-judging

*To resume* a killed run, just re-run the same command.  Finished steps are skipped automatically.
*To re-judge* with a newer rubric or judge model:

```bash
python3 longform_writing_bench.py \
  --test-model  "openai/gpt-4o" \
  --judge-model "anthropic/claude-3.7-sonnet" \
  --runs-file   "results/longform_bench_runs.json" \
  --run-id      "demo" \
  --skip-generation \
  --redo-judging
```

---

## Repository layout (key files only)

```
.
├─ longform_writing_bench.py      # CLI entry point
├─ core/
│   ├─ benchmark.py               # orchestration logic
│   ├─ conversation.py            # generation & judging task object
│   ├─ scoring.py                 # parsing, weighting, bootstrap
│   └─ metrics.py                 # auxiliary text metrics (slop, repetition, complexity)
├─ utils/
│   ├─ api.py                     # thin wrapper over HTTP calls with retry logic
│   ├─ file_io.py                 # atomic JSON read/write helpers
│   └─ logging_setup.py
├─ data/                          # prompt templates, criteria, slop lists, etc.
└─ results/                       # run logs (created at runtime)
```

---

## Customising the benchmark

* **Number of chapters** – set `NUM_CHAPTERS` in your environment before running.
* **Prompt templates** – edit `data/prompt*.txt`.  If you change the count, adjust `NUM_PLANNING_STEPS`.
* **Criteria weights** – see `data/criteria_weights.json`; unknown keys default to 1 × weight.
* **Negative metrics** – any criterion in `data/longform_negative_criteria_*.txt` is automatically inverted.

---

## Dependencies & data downloads

Python 3.9+ recommended.  All runtime deps are in `requirements.txt`.
After install, run once:

```python
import nltk
nltk.download("punkt")
nltk.download("cmudict")
nltk.download("stopwords")
```

Those are needed for `core/metrics.py`.  If you skip metrics, the main pipeline still works.

---

## Troubleshooting

* **“No tasks were successfully scored.”**
  Check judge responses in `results/…/longform_tasks.*.final_raw_judge_texts` – scores may be missing or mis-parsed.

* **Rate-limited (429).**
  The runner backs off exponentially.  Increase `RETRY_DELAY` or request a higher quota.

* **Long story cuts off early.**
  Bump `max_tokens` inside `LongformCreativeTask.run_generation_sequence` for chapter steps.

* **Threads overwhelm your API provider.**
  Lower `--threads`.  Generation is CPU-light; I/O wait dominates.

---

## License

MIT.  Attribution appreciated but not required.

---

## Citation

If this benchmark contributed to your research, cite it as:

```bibtex
@misc{paech2025longform,
  author = {Paech, S.J.},
  title = {Longform Creative Writing Benchmark},
  year = {2025},
  url = {https://github.com/EQ-bench/longform-writing-bench},
  note = {GitHub repository}
}
```
