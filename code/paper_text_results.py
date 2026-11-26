"""Export the numerical comparisons and scoring diagnostics quoted in the paper."""

from pathlib import Path

import pandas as pd

from offline_llm import read_json, read_lines
from paper_main_table import direct_gpa_share


def write_text_results(panel, metadata, estimates, reduced_forms, output_dir):
    """Write unrounded values; the column named group identifies the arm or proxy."""
    rows = []

    def add(statistic, group, value, unit):
        rows.append((statistic, group, value, unit))

    stages = {row["stage"]: row for row in metadata["sample_flow"]}
    eligible = stages["nonhonors_AI"]
    for treatment, arm_name in [(0, "Base"), (1, "Tutor")]:
        arm = panel.loc[panel["T"] == treatment]
        add("estimation_sample_sessions", arm_name, len(arm), "student-sessions")
        add("estimation_sample_students", arm_name, arm["student_id"].nunique(), "students")
        add("estimation_sample_classrooms", arm_name, arm["class_id"].nunique(), "classrooms")
        retained_pct = 100 * len(arm) / eligible[f"{arm_name}_sessions"]
        add("estimation_sample_retention", arm_name, retained_pct, "percent of non-honors AI sessions")
        add("unassisted_exam_mean", arm_name, 100 * arm["Y"].mean(), "percent of maximum score")
        add("observed_problems_mean", arm_name, arm["O_count"].mean(), "problems per student-session")
        add("understanding_share_mean", arm_name, 100 * arm["E_baseline"].mean(),
            "percent of classifiable problems, equal session weights")

    outcome = "Unassisted exam"
    base_score = panel.loc[panel["T"] == 0, "Y"].mean()
    score_sd = panel["Y"].std()
    score_model = estimates.iloc[0]  # All proxies share the same outcome regression.
    add("score_standard_deviation", outcome, score_sd, "fraction of maximum score")
    add("gpa_standard_deviation", "Prior GPA", panel["G"].std(), "fraction of maximum GPA")
    add("score_change_per_0.1_gpa", outcome, 10 * score_model["piYG"], "exam percentage points")
    add("adjusted_tutor_minus_base", outcome, 100 * score_model["piYT"], "exam percentage points")
    add("adjusted_tutor_minus_base_se", outcome, 100 * score_model["total_effect_se_CR1"],
        "exam percentage points")
    add("adjusted_tutor_minus_base_p", outcome, score_model["total_effect_p_t"], "two-sided p-value")

    base_mean = panel.loc[panel["T"] == 0].mean(numeric_only=True)
    tutor_mean = panel.loc[panel["T"] == 1].mean(numeric_only=True)
    add("raw_tutor_minus_base", "E_baseline",
        100 * (tutor_mean["E_baseline"] - base_mean["E_baseline"]), "percentage points")
    for proxy in ("E_baseline", "E_learning_observed"):
        add("tutor_to_base_raw_mean_ratio", proxy, tutor_mean[proxy] / base_mean[proxy],
            "ratio of student-session means")

    thresholds = direct_gpa_share(estimates)
    for index, row in estimates.iterrows():
        proxy = row["measure_id"]
        if proxy in ("E_baseline", "E_reasoning_observed"):
            add("adjusted_tutor_minus_base", proxy, 100 * row["theta_estimate"], "percentage points")
            add("adjusted_tutor_minus_base_se", proxy, 100 * row["theta_se_CR2"], "percentage points")
            add("proxy_change_per_0.1_gpa", proxy, 10 * row["piEG"], "percentage points")
        else:
            add("adjusted_tutor_minus_base", proxy, row["theta_estimate"], "proxy units")
            add("adjusted_tutor_minus_base_se", proxy, row["theta_se_CR2"], "proxy units")
            add("proxy_change_per_0.1_gpa", proxy, 0.1 * row["piEG"], "proxy units")
        add("return_per_proxy_sd", proxy, row["rho_estimate"] * panel[proxy].std() / score_sd,
            "exam standard deviations")
        add("effort_contribution", proxy, 100 * row["mediated_estimate"], "exam percentage points")
        add("effort_contribution_se", proxy, 100 * row["mediated_se_CR1"], "exam percentage points")
        add("effort_contribution_in_exam_sd", proxy, row["mediated_estimate"] / score_sd,
            "exam standard deviations")
        add("effort_contribution_relative_to_base_mean", proxy, row["mediated_estimate"] / base_score,
            "fraction of Base mean score")
        add("effect_beyond_effort", proxy, 100 * row["gamma_estimate"], "exam percentage points")
        add("effect_beyond_effort_se", proxy, 100 * row["gamma_se_CR1"], "exam percentage points")
        add("effect_beyond_effort_ci95_low", proxy, 100 * row["gamma_ci95_low"], "exam percentage points")
        add("effect_beyond_effort_ci95_high", proxy, 100 * row["gamma_ci95_high"], "exam percentage points")
        add("effect_beyond_effort_p", proxy, row["gamma_p_t"], "two-sided p-value")
        add("direct_gpa_share_at_zero_residual", proxy, 100 * thresholds.loc[index],
            "percent of reduced-form GPA coefficient")
        add("return_at_zero_residual", proxy, row["piYT"] / row["theta_estimate"],
            "fraction of maximum exam score per proxy unit")

    score_terms = reduced_forms.loc[reduced_forms["equation"] == "RF_Y", "term"].unique()
    add("outcome_indicator_count", outcome,
        len(set(score_terms) - {"(Intercept)", "G", "D", "T"}), "indicators")
    for statistic, group, field, unit in [
        ("design_rank", "Effort proxy", "rank_RF_E", "parameters"),
        ("design_rank", outcome, "rank_RF_Y", "parameters"),
        ("cr1_finite_sample_multiplier", "Effort proxy", "CR1_multiplier_RF_E", "multiplier"),
        ("cr1_finite_sample_multiplier", outcome, "CR1_multiplier_RF_Y", "multiplier"),
        ("cr1_reference_df", outcome, "df_CR1", "degrees of freedom"),
        ("theta_satterthwaite_df", "Effort proxy", "theta_df_CR2", "degrees of freedom"),
    ]:
        add(statistic, group, score_model[field], unit)

    archive = Path(__file__).resolve().parents[1] / "data" / "llm" / "scoring_run"
    llm = metadata["llm"]
    verified = llm["verification"]
    expanded = llm["expanded_learning_score_counts"]
    records = sum(expanded.values())
    repeated = verified["repeat_agreement"]["learning_score"]
    scoring_rows = [
        ("first_message_records", records, "student-problem records"),
        ("uninformative_first_messages", 100 * expanded.get("null", 0) / records, "percent"),
        ("unique_primary_inputs", verified["primary_valid"], "inputs"),
        ("primary_batches", verified["primary_batches_valid"], "batches"),
        ("batch_capacity", read_json(archive / "manifest.json")["batch_size"], "inputs per batch"),
        ("selected_messages_after_assistant_in_file_order",
         sum(item["after_assistant"] for item in read_lines(archive / "full_inputs.jsonl")), "records"),
        ("repeat_batches", verified["repeat_batches_valid"], "batches"),
        ("repeat_learning_scores", repeated["n"], "paired ratings"),
        ("repeat_learning_scores_agree", repeated["equal"], "paired ratings"),
        ("repeat_learning_score_agreement", 100 * repeated["fraction"], "percent"),
        ("invalid_scoring_attempts", verified["attempt_count"] - verified["valid_attempts"], "attempts"),
    ]
    for statistic, value, unit in scoring_rows:
        add(statistic, "Semantic scoring", value, unit)

    pd.DataFrame(rows, columns=["statistic", "group", "value", "unit"]).to_csv(
        Path(output_dir) / "paper_text_results.csv", index=False, lineterminator="\n")
