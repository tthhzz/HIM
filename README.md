# HIM: A Human-Inspired Memory Loop for LLM Agents

Official research snapshot for **“HIM: A Human-Inspired Memory Loop for LLM Agents via Encoding, Consolidation, and Retrieval.”**

Haoze Tang, Linqi Ye, and Shaorong Xie<br>
School of Future Technology, Shanghai University

[Read the paper](paper/HIM.pdf) · [Inspect archived results](results/README.md) · [Read the reproducibility audit](REPRODUCIBILITY.md)

## Overview

HIM is a reliability-oriented memory framework organized around a three-stage lifecycle:

1. **Encoding** binds speaker/source cues to structured notes and assigns importance-based STM, MTM, or LTM retention levels.
2. **Consolidation** uses retrieval-frequency signals to reinforce repeatedly accessed memories.
3. **Retrieval** preserves source attribution and records an ACT-R-inspired activation score built from semantic rank, usage, importance, and retention level.

The code in this repository is the manuscript-era experimental implementation. It is provided for traceability and may require model- or backend-specific adjustments.

## Repository layout

```text
.
├── paper/
│   └── HIM.pdf
├── src/
│   ├── him_memory.py          # HIM memory lifecycle implementation
│   ├── dataset.py             # LoCoMo data loader
│   └── metrics.py             # evaluation metrics
├── experiments/
│   ├── evaluate_him.py
│   ├── run_ablation.ps1
│   ├── run_ablation.bat
│   └── analyze_results.py
├── results/
│   ├── main/                  # selected per-question result artifacts
│   ├── paper_metrics.csv      # values transcribed from Tables I and II
│   └── README.md
├── data/
│   └── locomo10.json
├── REPRODUCIBILITY.md
├── requirements.txt
└── LICENSE
```

Generated logs, memory caches, ad-hoc experiment outputs, bytecode, credentials, and draft figures are intentionally excluded from Git.

## Environment

The original experiments used Python with local or OpenAI-compatible LLM backends. A minimal setup is:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For the Ollama configuration used by the archived Qwen2.5-3B runs, install and start Ollama separately and make the model available as `qwen2.5:3b`. OpenAI-compatible backends read credentials from `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`; never commit these values.

## Evaluation

Run HIM on the same 10% sampling setting recorded by the selected HIM artifact:

```bash
python experiments/evaluate_him.py \
  --dataset data/locomo10.json \
  --model qwen2.5:3b \
  --backend ollama \
  --ratio 0.1 \
  --retrieve_k 10 \
  --output results/runs/him_qwen2.5-3b_10pct.json
```

On Windows, the paper-oriented ablation configurations can be launched with either:

```powershell
.\experiments\run_ablation.ps1
```

or `experiments\run_ablation.bat`.

Summarize the archived HIM result file without running an LLM:

```bash
python experiments/analyze_results.py
```

## Archived paper results

The selected artifact reproduces the HIM row from Table I (F1 / BLEU-1, %):

| Method | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial |
|---|---:|---:|---:|---:|---:|
| **HIM** | **14.21 / 9.91** | **24.64 / 18.33** | **5.45 / 9.06** | **22.32 / 18.18** | **39.98 / 37.23** |

Important: the final ablation prediction files were not found, and the Table II prose contains two metric-label inconsistencies. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full audit before citing the numbers.

## License

The code is released under the MIT License. See [LICENSE](LICENSE). The paper and third-party dataset may be subject to their respective publication or dataset terms.

## Citation

If you use this snapshot, cite the manuscript:

```bibtex
@misc{tang2026him,
  title  = {HIM: A Human-Inspired Memory Loop for LLM Agents via Encoding, Consolidation, and Retrieval},
  author = {Tang, Haoze and Ye, Linqi and Xie, Shaorong},
  year   = {2026},
  note   = {Manuscript}
}
```

Replace this entry with the final venue citation if a published version is available.
