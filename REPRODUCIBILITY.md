# Reproducibility notes and repository audit

This repository is a research snapshot prepared from the working directory used for the HIM manuscript. The implementation and archived outputs are retained for traceability; they should not be interpreted as a polished, independently reproduced benchmark package.

## Archived evidence

- `results/main/him_qwen2.5-3b_10pct.json` contains the HIM predictions that reproduce the HIM row in Table I. It has 199 questions.
- `results/paper_metrics.csv` transcribes the final values printed in Tables I and II.
- Exact raw prediction files corresponding to the two ablations in the final Table II were not present in the reviewed working directory. Older ablation outputs were excluded because their aggregate values do not match the final table.

## Known limitations

1. **Table II and the accompanying prose disagree.** Table II reports `30.88` as Adversarial F1 for HIM without consolidation, while the paragraph cites `22.37`; `22.37` is that row's Temporal F1. The paragraph also cites `16.69` as Temporal F1, while Table II reports `22.37`; `16.69` is Temporal BLEU-1. The CSV follows the table itself.
2. **Ablation artifacts are incomplete.** The final w/o Encoding and w/o Consolidation raw predictions could not be identified among the available JSON files, so only the printed table values are archived here.
3. **Consolidation timing is implementation-sensitive.** In the archived evaluator, periodic consolidation is triggered while memories are ingested, whereas retrieval counts are updated later during question answering. In a fresh run, early consolidation calls may therefore see no retrieval usage. This should be revisited before claiming a clean causal ablation.
4. **Activation is diagnostic in this snapshot.** The activation score is logged for interpretation and usage accounting; semantic similarity remains the primary retrieval ordering, matching the implementation comments.

## Public-repository hygiene

- Cache directories, logs, Python bytecode, ad-hoc result files, and connectivity tests are ignored.
- Absolute local dataset paths were removed from the two published JSON artifacts.
- The public copy of the paper has sanitized PDF metadata; the manuscript content is unchanged.
- Standalone draft case-study figures were omitted because none exactly matched the figure embedded in the final PDF.

These notes describe the reviewed snapshot, not any later corrected experiment run.
