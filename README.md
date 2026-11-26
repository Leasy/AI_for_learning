# AI tutoring and learning

Data preparation and estimation for GPT Tutor versus GPT Base using the released data from Bastani et al. (2025), *Generative AI without guardrails can harm learning: Evidence from high school mathematics*.

## Model

The indices denote student i, classroom c and session s.

```text
E_ics = pi_0 + theta * T_c + pi_G * G_i + pi_D * D_cs + xi_ics
Y_ics = delta_0 + rho * E_ics + gamma * T_c + delta_D * D_cs + delta_W' * W_ics + eta_ics
```

`E`: observed proxy for effective effort; `G`: prior GPA; `T`: GPT Tutor (1) versus GPT Base (0); `D`: difficulty; `Y`: normalized unaided exam score. `W` contains grade, session, teacher and grader indicators. Difficulty uses non-honors no-AI control-arm practice scores. `theta` is the safeguard-induced effort response in proxy units; `rho` is the score return to effort per proxy unit; `gamma` is the score effect beyond effort.

The estimated reduced forms are:

```text
E_ics = pi_E0 + pi_ET * T_c + pi_EG * G_i + pi_ED * D_cs + error_E_ics
Y_ics = pi_Y0 + pi_YT * T_c + pi_YG * G_i + pi_YD * D_cs + pi_YW' * W_ics + error_Y_ics

theta = pi_ET
rho = pi_YG / pi_EG
gamma = pi_YT - rho * theta
```

Regressors: `E ~ 1 + T + D + G`; `Y ~ 1 + T + D + G + W`. Estimation retains the restricted four-column first stage and the separate outcome reduced form. The IV representation uses prior GPA `G` to instrument `E`; an unrestricted standard 2SLS fit would also include `W` in its first stage and can differ in finite samples. The sample contains 1,163 student-sessions, 468 students and 27 classrooms, with equal student-session weights.

Maintained assumptions are linear equations; the same positive effort-to-proxy measurement relation across arms, with no direct treatment effect on measurement at fixed effort; conditional GPA relevance; GPA exclusion from scores except through effective effort; and `Cov(Q,xi)=Cov(Q,eta)=0` for `Q=(T,G,D,W)`. `E` may correlate with `eta`, and the two errors may correlate with one another. The hypotheses are `theta>0`, `rho>0`, and `gamma=0`; the last tests for an additional net score effect under the model.

## Dialogue measures

`O`: problems with a student message; `U`: understanding evidence; `A`: answer-only evidence. Rule-based labels use all student messages per problem; semantic scores use the first student message.

| CSV column | Definition |
| --- | --- |
| `E_baseline` | `U/(U+A)`; mixed problems count in `U`. |
| `E_learning_observed` | `sum(non-null learning_score)/(3*O)`; first-message semantic scores range from 0 to 3. |
| `E_net_observed` | `(U_only-A_only)/O`; mixed and unclassified problems contribute zero. |
| `E_reasoning_observed` | `R/O`; specific process help, reasoning, attempts, diagnosis, checks, or help without an answer. |

`R` uses `specific_why_question`, `specific_how_step`, `specific_step_explanation`, `articulated_reasoning`, `described_attempt`, `error_diagnosis`, `reasoning_check`, and `requests_help_without_answer`.

## Data

Released inputs and hashes: `data/input_manifest.json`. Semantic-scoring prompts, responses, linkage and provenance: `data/llm/scoring_run/`. Cleaning reconstructs the dialogue measures and verifies the saved scoring artifacts.

## Inference

Classroom clustering; CR2/Satterthwaite for `theta`; joint CR1 delta covariance and `t(26)` for `rho` and `gamma`; 95% Fieller sets for `rho`. Each equation uses its own CR1 finite-sample correction. Displayed p-values are two-sided. Holm fields cover the three alternative-measure theta tests.

## Reproduction

On Windows, double-click [run_all.cmd](run_all.cmd). The launcher uses Python 3.10+ with NumPy and pandas, and base R 4.0+ (Rscript). It can be launched from any working directory. From a terminal the same entry point is:

```powershell
python -B -X utf8 run_all.py
```

The run reconstructs the analysis sample and dialogue scores from the released data and archived model responses, fits the four specifications in R, and writes the publication tables and in-text statistics. The manuscript reads the five LaTeX tables directly from [results/tables/](results/tables/).

Open the [results index](results/README.md) after the run. Each numbered table has a LaTeX file and a CSV with the same displayed values and paper column order.

| Paper exhibit | Subject | Output in `results/tables/` |
| --- | --- | --- |
| Table 1 | Construction of the estimation sample | `table_1_sample.{tex,csv}` |
| Table 2 | Summary statistics | `table_2_summary.{tex,csv}` |
| Table 3 | Effort, GPA instrument and exam-score decomposition, including the direct GPA share at zero residual | `table_3_main.{tex,csv}` |
| Table A1 | Semantic scoring rubric | `table_a1_rubric.{tex,csv}` |
| Table A2 | First-message scores and their distribution | `table_a2_scores.{tex,csv}` |

- [paper_text_results.csv](results/paper_text_results.csv) gives the computed statistics quoted in the text, with a statistic name, group or proxy, value, and unit.
- [main_table.md](results/main_table.md) displays Table 3. The top-level `main_table.csv` and `main_table.tex` are copies of the numbered Table 3.
- [main_estimates.csv](results/main_estimates.csv) contains full-precision estimates and inference. [reduced_form_coefficients.csv](results/reduced_form_coefficients.csv) gives all 152 reduced-form coefficient rows. [regression_diagnostics.md](results/regression_diagnostics.md) presents the detailed inference output, also supplied as CSV and LaTeX.
- `regression_info.json` and `run_info.json` give the regression checks, sample counts, file locations, and run status.

To use another output location, pass `--clean-output PATH.csv` and `--output-dir PATH` to the launcher or Python entry point. `--rscript PATH` selects an R installation. For development, `python -B -X utf8 code/clean_data.py` runs the data stage and `Rscript --vanilla code/run_regressions.R` runs the regression stage.
