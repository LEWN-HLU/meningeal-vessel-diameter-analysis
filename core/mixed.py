# Vessel imaging analysis methods.
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels
from patsy import build_design_matrices
from scipy import stats
from core.stats import add_holm

def mixed_model_size_interaction(
    animal_df: pd.DataFrame,
    reference: str,
    comparison_groups: list[str],
    holm_across_sizes: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Test treatment-by-size interaction and planned within-size contrasts."""
    omnibus_columns = [
        "Test",
        "Wald chi-square",
        "df",
        "p-value",
        "Observations",
        "Mouse clusters",
        "Converged",
    ]
    contrast_columns = [
        "Comparison",
        "Vessel size",
        "Estimate",
        "SE",
        "z",
        "Raw p",
        "Holm-adjusted p",
        "Significance",
    ]
    work = animal_df.copy().dropna(
        subset=["z_response", "group", "vessel_size", "mouse_alias", "dataset_id"]
    )
    work = work[work["vessel_size"].astype(str).isin(["Small", "Large"])].copy()
    available_groups = set(work["group"].astype(str))
    comparison_groups = [
        str(group)
        for group in comparison_groups
        if str(group) in available_groups and str(group) != str(reference)
    ]
    if work.empty or str(reference) not in available_groups or not comparison_groups:
        return (
            pd.DataFrame(columns=omnibus_columns),
            pd.DataFrame(columns=contrast_columns),
            "No complete treatment, vehicle, Small, and Large observations were available.",
        )

    work["group"] = work["group"].astype(str)
    work["vessel_size"] = work["vessel_size"].astype(str)
    work["mouse_cluster"] = work["mouse_alias"].astype(str)
    blank = work["mouse_cluster"].isin(["", "nan", "None"])
    work.loc[blank, "mouse_cluster"] = work.loc[blank, "dataset_id"].astype(str)
    formula = (
        f"z_response ~ C(group, Treatment(reference={str(reference)!r})) * "
        "C(vessel_size, Treatment(reference='Small'))"
    )

    fit = None
    errors: list[str] = []
    for method in ["lbfgs", "powell"]:
        try:
            model = smf.mixedlm(formula, data=work, groups=work["mouse_cluster"])
            candidate = model.fit(reml=False, method=method, maxiter=1000, disp=False)
            fit = candidate
            if bool(getattr(candidate, "converged", False)):
                break
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    if fit is None:
        return (
            pd.DataFrame(columns=omnibus_columns),
            pd.DataFrame(columns=contrast_columns),
            "Mixed-effects model could not be estimated. " + " | ".join(errors),
        )

    try:
        design_info = fit.model.data.design_info
        fe_names = list(fit.fe_params.index)
        beta = fit.fe_params.to_numpy(float)
        covariance = fit.cov_params().loc[fe_names, fe_names].to_numpy(float)
        interaction_indices = [
            index
            for index, name in enumerate(fe_names)
            if ":" in name and "C(group" in name and "C(vessel_size" in name
        ]
        if not interaction_indices:
            raise ValueError("No treatment-by-vessel-size interaction coefficients were estimable.")

        interaction_beta = beta[interaction_indices]
        interaction_covariance = covariance[np.ix_(interaction_indices, interaction_indices)]
        interaction_df = int(np.linalg.matrix_rank(interaction_covariance))
        interaction_statistic = float(
            interaction_beta @ np.linalg.pinv(interaction_covariance) @ interaction_beta
        )
        interaction_p = (
            float(stats.chi2.sf(interaction_statistic, interaction_df))
            if interaction_df > 0
            else np.nan
        )
        omnibus = pd.DataFrame(
            [
                {
                    "Test": "Treatment x vessel-size interaction",
                    "Wald chi-square": interaction_statistic,
                    "df": interaction_df,
                    "p-value": interaction_p,
                    "Observations": len(work),
                    "Mouse clusters": work["mouse_cluster"].nunique(),
                    "Converged": bool(getattr(fit, "converged", False)),
                }
            ],
            columns=omnibus_columns,
        )

        contrast_rows = []
        for vessel_size in ["Small", "Large"]:
            for group in comparison_groups:
                new = pd.DataFrame(
                    {
                        "group": [str(group), str(reference)],
                        "vessel_size": [vessel_size, vessel_size],
                    }
                )
                matrix = np.asarray(build_design_matrices([design_info], new)[0], dtype=float)
                contrast = matrix[0] - matrix[1]
                estimate = float(contrast @ beta)
                variance = float(contrast @ covariance @ contrast)
                se = float(np.sqrt(max(variance, 0)))
                z_value = estimate / se if se > 0 else np.nan
                p_value = 2 * stats.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
                contrast_rows.append(
                    {
                        "Comparison": f"{group} vs {reference}",
                        "Vessel size": vessel_size,
                        "Estimate": estimate,
                        "SE": se,
                        "z": z_value,
                        "Raw p": p_value,
                    }
                )
        contrast_frame = pd.DataFrame(contrast_rows)
        if holm_across_sizes:
            contrasts = add_holm(contrast_frame)
            holm_note = (
                "Holm correction is applied jointly across the Small and Large "
                "follow-up contrasts within this study."
            )
        else:
            contrasts = pd.concat(
                [
                    add_holm(size_frame)
                    for _, size_frame in contrast_frame.groupby(
                        "Vessel size", observed=True, sort=False
                    )
                ],
                ignore_index=True,
            )
            holm_note = (
                "Holm correction is applied separately to each vessel-size set "
                "within this study."
            )
        contrasts = contrasts[contrast_columns]
        note = (
            "Linear mixed-effects model fitted by maximum likelihood: z_response ~ treatment * "
            "vessel_size + (1 | mouse_alias). The treatment-by-size Wald test is the primary test. "
            "Planned challenge-versus-vehicle contrasts are estimated within Small and Large vessels. "
            f"{holm_note} "
            f"Observations={len(work)}; mouse clusters={work['mouse_cluster'].nunique()}; "
            f"converged={bool(getattr(fit, 'converged', False))}."
        )
        return omnibus, contrasts, note
    except Exception as exc:
        return (
            pd.DataFrame(columns=omnibus_columns),
            pd.DataFrame(columns=contrast_columns),
            f"Mixed-effects model fit was returned, but inference could not be calculated: {exc}",
        )

def _mouse_level_two_group_values(
    animal_df: pd.DataFrame,
    reference: str,
    comparison: str,
    category_col: str | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    required = ["z_response", "group", "mouse_alias"]
    if category_col is not None:
        required.append(category_col)
    work = animal_df.copy().dropna(subset=required)
    work["group"] = work["group"].astype(str)
    work["mouse_alias"] = work["mouse_alias"].astype(str)
    work = work[work["group"].isin([str(reference), str(comparison)])].copy()
    if category_col is not None:
        work[category_col] = work[category_col].astype(str)
        if categories is not None:
            work = work[work[category_col].isin([str(value) for value in categories])].copy()
    group_cols = ["mouse_alias", "group"] + ([category_col] if category_col is not None else [])
    return (
        work.groupby(group_cols, observed=True, as_index=False)
        .agg(
            z_response=("z_response", "mean"),
            source_sessions=("dataset_id", "nunique") if "dataset_id" in work.columns else ("z_response", "size"),
            vessel_contributions=("vessels", "sum") if "vessels" in work.columns else ("z_response", "size"),
        )
    )

def _two_group_coverage(
    work: pd.DataFrame,
    reference: str,
    comparison: str,
) -> tuple[int, int, int]:
    presence = (
        work.assign(present=True)
        .pivot_table(
            index="mouse_alias",
            columns="group",
            values="present",
            aggfunc="any",
            fill_value=False,
        )
        .reindex(columns=[str(reference), str(comparison)], fill_value=False)
    )
    matched = int((presence[str(reference)] & presence[str(comparison)]).sum())
    reference_only = int((presence[str(reference)] & ~presence[str(comparison)]).sum())
    comparison_only = int((~presence[str(reference)] & presence[str(comparison)]).sum())
    return matched, reference_only, comparison_only

def _fit_random_mouse_intercept(formula: str, work: pd.DataFrame):
    fit = None
    optimizer = ""
    errors: list[str] = []
    for method in ["lbfgs", "powell"]:
        try:
            candidate = smf.mixedlm(
                formula,
                data=work,
                groups=work["mouse_alias"],
            ).fit(reml=False, method=method, maxiter=1000, disp=False)
            fit = candidate
            optimizer = method
            if bool(getattr(candidate, "converged", False)):
                break
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    return fit, optimizer, errors

def mixed_model_two_group_mouse_intercept(
    animal_df: pd.DataFrame,
    reference: str,
    comparison: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Compare two partially overlapping treatment groups with a random mouse intercept."""
    result_columns = [
        "Comparison",
        "Estimate",
        "SE",
        "z",
        "Raw p",
        "Holm-adjusted p",
        "Significance",
        "Reference mice",
        "Comparison mice",
        "Matched mice",
        "Reference-only mice",
        "Comparison-only mice",
    ]
    work = _mouse_level_two_group_values(animal_df, reference, comparison)
    groups = set(work["group"].astype(str)) if not work.empty else set()
    if work.empty or not {str(reference), str(comparison)}.issubset(groups):
        note = "Both PACAP and Anti-PACAP mAb require at least one mouse-level value."
        return work, pd.DataFrame(columns=result_columns), pd.DataFrame(), note
    formula = f"z_response ~ C(group, Treatment(reference={str(reference)!r}))"
    fit, optimizer, errors = _fit_random_mouse_intercept(formula, work)
    matched, reference_only, comparison_only = _two_group_coverage(work, reference, comparison)
    if fit is None:
        note = "Mixed-effects model could not be estimated. " + " | ".join(errors)
        return work, pd.DataFrame(columns=result_columns), pd.DataFrame(), note
    try:
        design_info = fit.model.data.design_info
        fe_names = list(fit.fe_params.index)
        beta = fit.fe_params.to_numpy(float)
        covariance = fit.cov_params().loc[fe_names, fe_names].to_numpy(float)
        new = pd.DataFrame({"group": [str(comparison), str(reference)]})
        matrix = np.asarray(build_design_matrices([design_info], new)[0], dtype=float)
        contrast = matrix[0] - matrix[1]
        estimate = float(contrast @ beta)
        variance = float(contrast @ covariance @ contrast)
        se = float(np.sqrt(max(variance, 0)))
        z_value = estimate / se if se > 0 else np.nan
        p_value = 2 * stats.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
        results = add_holm(
            pd.DataFrame(
                [
                    {
                        "Comparison": f"{comparison} minus {reference}",
                        "Estimate": estimate,
                        "SE": se,
                        "z": z_value,
                        "Raw p": p_value,
                        "Reference mice": int(work.loc[work["group"] == str(reference), "mouse_alias"].nunique()),
                        "Comparison mice": int(work.loc[work["group"] == str(comparison), "mouse_alias"].nunique()),
                        "Matched mice": matched,
                        "Reference-only mice": reference_only,
                        "Comparison-only mice": comparison_only,
                    }
                ]
            )
        )[result_columns]
        random_variance = (
            float(np.asarray(fit.cov_re, dtype=float).ravel()[0])
            if np.asarray(fit.cov_re).size
            else np.nan
        )
        model_summary = pd.DataFrame(
            [
                {
                    "Model": "Two-group random-intercept mixed model",
                    "Formula": f"z_response ~ treatment + (1 | mouse_alias)",
                    "Software": f"statsmodels {statsmodels.__version__}",
                    "Observations": int(len(work)),
                    "Mouse clusters": int(work["mouse_alias"].nunique()),
                    "Matched mice": matched,
                    "Reference-only mice": reference_only,
                    "Comparison-only mice": comparison_only,
                    "Optimizer": optimizer,
                    "Converged": bool(getattr(fit, "converged", False)),
                    "Random-intercept variance": random_variance,
                    "Log likelihood": float(fit.llf),
                }
            ]
        )
        note = (
            f"Linear mixed-effects model fitted by maximum likelihood: z_response ~ treatment + "
            f"(1 | mouse_alias). Estimate is {comparison} minus {reference}; a negative estimate means "
            f"the Anti-PACAP response is lower. Observations are one mouse-level value per treatment. "
            f"Matched mice={matched}, {reference}-only={reference_only}, {comparison}-only={comparison_only}."
        )
        return work, results, model_summary, note
    except Exception as exc:
        return (
            work,
            pd.DataFrame(columns=result_columns),
            pd.DataFrame(),
            f"Mixed-effects model fit was returned, but inference could not be calculated: {exc}",
        )

def mixed_model_two_group_category_interaction(
    animal_df: pd.DataFrame,
    reference: str,
    comparison: str,
    category_col: str,
    categories: list[str],
    category_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Fit treatment-by-category mixed model and contrasts within each category."""
    contrast_columns = [
        "Comparison",
        category_label,
        "Estimate",
        "SE",
        "z",
        "Raw p",
        "Holm-adjusted p",
        "Significance",
        "Reference mice",
        "Comparison mice",
        "Matched mice",
        "Reference-only mice",
        "Comparison-only mice",
    ]
    work = _mouse_level_two_group_values(
        animal_df,
        reference,
        comparison,
        category_col=category_col,
        categories=categories,
    )
    valid_categories = [
        str(category)
        for category in categories
        if {str(reference), str(comparison)}.issubset(
            set(work.loc[work[category_col] == str(category), "group"].astype(str))
        )
    ]
    if len(valid_categories) < 2:
        note = f"Two categories with both treatment groups are required for the interaction model."
        return work, pd.DataFrame(), pd.DataFrame(columns=contrast_columns), pd.DataFrame(), note
    work = work[work[category_col].isin(valid_categories)].copy()
    category_reference = valid_categories[0]
    formula = (
        f"z_response ~ C(group, Treatment(reference={str(reference)!r})) * "
        f"C({category_col}, Treatment(reference={category_reference!r}))"
    )
    fit, optimizer, errors = _fit_random_mouse_intercept(formula, work)
    if fit is None:
        note = "Mixed-effects model could not be estimated. " + " | ".join(errors)
        return work, pd.DataFrame(), pd.DataFrame(columns=contrast_columns), pd.DataFrame(), note
    try:
        design_info = fit.model.data.design_info
        fe_names = list(fit.fe_params.index)
        beta = fit.fe_params.to_numpy(float)
        covariance = fit.cov_params().loc[fe_names, fe_names].to_numpy(float)
        interaction_indices = [
            index
            for index, name in enumerate(fe_names)
            if ":" in name and "C(group" in name and f"C({category_col}" in name
        ]
        interaction_beta = beta[interaction_indices]
        interaction_covariance = covariance[np.ix_(interaction_indices, interaction_indices)]
        interaction_df = int(np.linalg.matrix_rank(interaction_covariance))
        interaction_statistic = (
            float(interaction_beta @ np.linalg.pinv(interaction_covariance) @ interaction_beta)
            if interaction_indices
            else np.nan
        )
        interaction_p = (
            float(stats.chi2.sf(interaction_statistic, interaction_df))
            if interaction_df > 0 and np.isfinite(interaction_statistic)
            else np.nan
        )
        omnibus = pd.DataFrame(
            [
                {
                    "Test": f"Treatment x {category_label.lower()} interaction",
                    "Wald chi-square": interaction_statistic,
                    "df": interaction_df,
                    "p-value": interaction_p,
                    "Observations": int(len(work)),
                    "Mouse clusters": int(work["mouse_alias"].nunique()),
                    "Converged": bool(getattr(fit, "converged", False)),
                }
            ]
        )
        contrast_rows: list[dict] = []
        for category in valid_categories:
            category_work = work[work[category_col] == category]
            matched, reference_only, comparison_only = _two_group_coverage(
                category_work, reference, comparison
            )
            new = pd.DataFrame(
                {
                    "group": [str(comparison), str(reference)],
                    category_col: [category, category],
                }
            )
            matrix = np.asarray(build_design_matrices([design_info], new)[0], dtype=float)
            contrast = matrix[0] - matrix[1]
            estimate = float(contrast @ beta)
            variance = float(contrast @ covariance @ contrast)
            se = float(np.sqrt(max(variance, 0)))
            z_value = estimate / se if se > 0 else np.nan
            p_value = 2 * stats.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
            contrast_rows.append(
                {
                    "Comparison": f"{comparison} minus {reference}",
                    category_label: category,
                    "Estimate": estimate,
                    "SE": se,
                    "z": z_value,
                    "Raw p": p_value,
                    "Reference mice": int(category_work.loc[category_work["group"] == str(reference), "mouse_alias"].nunique()),
                    "Comparison mice": int(category_work.loc[category_work["group"] == str(comparison), "mouse_alias"].nunique()),
                    "Matched mice": matched,
                    "Reference-only mice": reference_only,
                    "Comparison-only mice": comparison_only,
                }
            )
        contrasts = add_holm(pd.DataFrame(contrast_rows))[contrast_columns]
        random_variance = (
            float(np.asarray(fit.cov_re, dtype=float).ravel()[0])
            if np.asarray(fit.cov_re).size
            else np.nan
        )
        model_summary = pd.DataFrame(
            [
                {
                    "Model": f"Treatment-by-{category_label.lower()} random-intercept mixed model",
                    "Formula": f"z_response ~ treatment * {category_col} + (1 | mouse_alias)",
                    "Software": f"statsmodels {statsmodels.__version__}",
                    "Observations": int(len(work)),
                    "Mouse clusters": int(work["mouse_alias"].nunique()),
                    "Optimizer": optimizer,
                    "Converged": bool(getattr(fit, "converged", False)),
                    "Random-intercept variance": random_variance,
                    "Log likelihood": float(fit.llf),
                }
            ]
        )
        note = (
            f"Linear mixed-effects model fitted by maximum likelihood: z_response ~ treatment * "
            f"{category_col} + (1 | mouse_alias). The interaction asks whether the treatment difference "
            f"changes between {', '.join(valid_categories)}. Planned {comparison}-minus-{reference} contrasts "
            f"are reported within each {category_label.lower()} and Holm-adjusted together across those "
            f"{len(valid_categories)} contrasts."
        )
        return work, omnibus, contrasts, model_summary, note
    except Exception as exc:
        return (
            work,
            pd.DataFrame(),
            pd.DataFrame(columns=contrast_columns),
            pd.DataFrame(),
            f"Mixed-effects model fit was returned, but inference could not be calculated: {exc}",
        )
