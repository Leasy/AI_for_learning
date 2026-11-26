"""Build the manuscript's three-panel regression table from the fitted models."""
import math
import re

from paper_table_io import render_table


PAPER_MEASURES = (
    "E_baseline", "E_net_observed", "E_reasoning_observed", "E_learning_observed",
)
PAPER_HEADERS = (
    "Understanding share", "Net understanding", "Specific reasoning", "Semantic score",
)


def direct_gpa_share(estimates):
    """Fraction of the score-GPA association required to bring gamma to zero."""
    effort_component = estimates["theta_estimate"] * estimates["rho_estimate"]
    return 1 - estimates["total_effect_estimate"] / effort_component.replace(0, float("nan"))


def _numbers(values, digits):
    return [f"{value:.{digits}f}" for value in values]


def _standard_errors(values, digits):
    return [f"({value:.{digits}f})" for value in values]


def _intervals(lower, upper, digits):
    return [f"[{lo:.{digits}f}, {hi:.{digits}f}]" for lo, hi in zip(lower, upper)]


def _fieller_cell(text):
    # Preserve the interval boundaries, including half-lines and disjoint sets.
    return re.sub(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
                  lambda match: f"{float(match[0]):.2f}", str(text))


def _latex_cell(text):
    text = re.sub(r"-\d+(?:\.\d+)?", lambda match: "$" + match[0] + "$", text)
    return text.replace("-Inf", r"$-\infty$").replace("Inf", r"$\infty$").replace(" union ", r" $\cup$ ")


def write_main_table(panel, estimates, reduced_forms, tables_dir):
    """Write the paper's Table 3 in its published column order."""
    fit = estimates.set_index("measure_id", verify_integrity=True).loc[list(PAPER_MEASURES)]
    gpa = reduced_forms.loc[reduced_forms["term"] == "G"]
    gpa = gpa.set_index(["equation", "measure_id"], verify_integrity=True)
    base_means = panel.loc[panel["T"].eq(0), list(PAPER_MEASURES)].mean()
    rows = []
    csv_rows = [["Panel", "Statistic", *PAPER_HEADERS]]

    def heading(title):
        if rows:
            rows.append(r"\addlinespace")
        rows.extend([r"\multicolumn{5}{@{}l}{\textit{" + title + r"}} \\", r"\addlinespace[2pt]"])

    def row(csv_label, label, values, shared=False):
        cells = list(values)
        csv_rows.append([section, csv_label, *cells])
        if shared:
            if len(set(cells)) != 1:
                raise ValueError(f"The common score equation differs across measures for {csv_label}")
            text = r"\multicolumn{4}{c}{" + _latex_cell(cells[0]) + "}"
        else:
            text = " & ".join(_latex_cell(cell) for cell in cells)
        rows.append(label + " & " + text + r" \\")

    section = "A"
    heading("Panel A. Effect of GPT Tutor on the effort proxy")
    row("Tutor coefficient", r"Tutor, $\hat\theta$", _numbers(fit["theta_estimate"], 3))
    row("Tutor coefficient SE", "", _standard_errors(fit["theta_se_CR2"], 3))
    row("Tutor coefficient 95% CI", "",
        _intervals(fit["theta_ci95_CR2_low"], fit["theta_ci95_CR2_high"], 3))
    row("Base-arm mean", "Base-arm mean", _numbers(base_means, 3))

    section = "B"
    heading("Panel B. Prior GPA as the excluded instrument")
    proxy_gpa = gpa.loc["RF_E"].loc[list(PAPER_MEASURES)]
    score_gpa = gpa.loc["RF_Y"].loc[list(PAPER_MEASURES)]
    row("GPA coefficient in proxy equation", r"GPA in proxy equation, $\hat\pi_{EG}$",
        _numbers(proxy_gpa["estimate"], 3))
    row("GPA coefficient in proxy equation SE", "", _standard_errors(proxy_gpa["se_CR1"], 3))
    row("GPA coefficient in score equation (SE)", r"GPA in score equation, $\hat\pi_{YG}$",
        [f"{b:.3f} ({se:.3f})" for b, se in zip(score_gpa["estimate"], score_gpa["se_CR1"])],
        shared=True)
    row("Score return", r"Return, $\hat\rho=\hat\pi_{YG}/\hat\pi_{EG}$",
        _numbers(fit["rho_estimate"], 2))
    row("Score return SE", "", _standard_errors(fit["rho_se_CR1"], 2))
    row("Fieller 95% confidence set", r"\quad Fieller 95 percent set",
        [_fieller_cell(text) for text in fit["rho_Fieller95_set_full"]])
    row("First-stage F-statistic", r"First-stage $F$-statistic", _numbers(fit["first_stage_F"], 1))

    section = "C"
    heading("Panel C. Decomposition of the Tutor--Base exam-score difference")
    row("Total score difference (SE)", r"Total, $\hat\pi_{YT}$",
        [f"{b:.3f} ({se:.3f})" for b, se in zip(fit["total_effect_estimate"], fit["total_effect_se_CR1"])],
        shared=True)
    row("Through the proxy", r"Through the proxy, $\hat\rho\hat\theta$", _numbers(fit["mediated_estimate"], 3))
    row("Through the proxy SE", "", _standard_errors(fit["mediated_se_CR1"], 3))
    row("Residual", r"Residual, $\hat\gamma$", _numbers(fit["gamma_estimate"], 3))
    row("Residual SE", "", _standard_errors(fit["gamma_se_CR1"], 3))
    row("Residual 95% CI", "", _intervals(fit["gamma_ci95_low"], fit["gamma_ci95_high"], 3))
    shares = direct_gpa_share(fit)
    row("Direct GPA share for gamma = 0", r"Direct GPA share for $\gamma=0$",
        [f"{share:.3f}" if math.isfinite(share) else "undefined" for share in shares])
    rows.append(r"\midrule")
    section = "Sample"
    row("Student-sessions", "Student-sessions", [f"{n:,d}" for n in fit["n"]])
    row("Classrooms", "Classrooms", [f"{n:d}" for n in fit["classrooms"]])
    if fit["theta_df_CR2"].nunique() != 1 or fit["df_CR1"].nunique() != 1:
        raise ValueError("Table note requires the common proxy design and classroom sample")
    fields = {"THETA_DF": f"{fit['theta_df_CR2'].iloc[0]:.1f}",
              "CLUSTER_DF": str(int(fit["df_CR1"].iloc[0]))}
    render_table(tables_dir, "table_3_main", rows, csv_rows, fields=fields)
