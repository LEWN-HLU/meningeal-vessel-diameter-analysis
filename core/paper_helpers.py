# Vessel imaging analysis methods.
from __future__ import annotations
import platform
import warnings
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from scipy import stats

COMPARTMENT_ORDER = ["dura", "pia"]

RESPONSE_WINDOW = "Response 1"

def canonical_mouse(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text

def with_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["original_study"] = (
        work["original_study"].fillna(work["study"])
        if "original_study" in work
        else work["study"]
    )
    work["original_treatment"] = (
        work["original_treatment"].fillna(work["group"])
        if "original_treatment" in work
        else work["group"]
    )
    return work

def response1(frame: pd.DataFrame) -> pd.DataFrame:
    if "window_order" in frame:
        return frame[pd.to_numeric(frame["window_order"], errors="coerce").eq(1)].copy()
    return frame[frame["window"].astype(str).eq(RESPONSE_WINDOW)].copy()

def session_values(
    frame: pd.DataFrame,
    category_col: str | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    work = response1(frame)
    keys = ["study", "group", "dataset_id", "mouse_alias"]
    if category_col:
        work[category_col] = work[category_col].astype(str)
        work = work[work[category_col].isin(categories or [])].copy()
        keys.append(category_col)
    return (
        work.groupby(keys, observed=True, as_index=False)
        .agg(z_response=("z_response", "mean"), vessels=("vessel_index", "nunique"))
    )

def mouse_compartment_values(frame: pd.DataFrame) -> pd.DataFrame:
    sessions = session_values(frame, "compartment", COMPARTMENT_ORDER)
    if sessions.empty:
        return sessions
    sessions["mouse_alias"] = sessions["mouse_alias"].map(canonical_mouse)
    return (
        sessions.groupby(["mouse_alias", "compartment"], observed=True, as_index=False)
        .agg(
            z_response=("z_response", "mean"),
            sessions=("dataset_id", "nunique"),
            vessels=("vessels", "sum"),
        )
    )

def fit_pacap_compartment(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    work = mouse_compartment_values(frame)
    if work.empty or not set(COMPARTMENT_ORDER).issubset(set(work["compartment"])):
        return work, pd.DataFrame(), "Both Dura and Pia mouse-level values are required."
    formula = "z_response ~ C(compartment, Treatment(reference='dura'))"
    fit = None
    errors: list[str] = []
    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = smf.mixedlm(formula, work, groups=work["mouse_alias"]).fit(
                    reml=False, method=method, maxiter=1000, disp=False
                )
            fit = candidate
            if bool(getattr(candidate, "converged", False)):
                break
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    if fit is None:
        return work, pd.DataFrame(), "Model could not be fitted: " + " | ".join(errors)
    term = [name for name in fit.fe_params.index if "compartment" in name][0]
    estimate = float(fit.fe_params[term])
    se = float(fit.bse_fe[term])
    z_value = estimate / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    dura = set(work.loc[work["compartment"].eq("dura"), "mouse_alias"])
    pia = set(work.loc[work["compartment"].eq("pia"), "mouse_alias"])
    result = pd.DataFrame([{
        "Comparison": "Pia minus Dura",
        "Estimate": estimate,
        "SE": se,
        "z": z_value,
        "p-value": p_value,
        "Dura mice": len(dura),
        "Pia mice": len(pia),
        "Mice with both": len(dura & pia),
        "Unique mice": len(dura | pia),
        "Observations": len(work),
        "Converged": bool(getattr(fit, "converged", False)),
        "Random-intercept variance": float(np.asarray(fit.cov_re).ravel()[0]),
    }])
    note = (
        "Maximum-likelihood linear mixed-effects model: z_response ~ compartment + "
        "(1 | mouse). The analysis uses one mean per mouse and compartment, retains "
        "partially matched mice, and reports Pia minus Dura."
    )
    return work, result, note

def fit_baseline_compartment(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = frame.copy()
    work["compartment"] = work["compartment"].fillna("unknown").astype(str).str.lower()
    work = work[work["compartment"].isin(COMPARTMENT_ORDER)].copy()
    work = work[np.isfinite(pd.to_numeric(work["baseline_um"], errors="coerce"))].copy()
    work["mouse_cluster"] = work["mouse_alias"].map(canonical_mouse)
    work["session_key"] = work["study"].astype(str) + " | " + work["group"].astype(str) + " | " + work["dataset_id"].astype(str)
    formula = "baseline_um ~ C(compartment, Treatment(reference='dura')) + C(paper_source)"
    fit = None
    optimizer = ""
    errors: list[str] = []
    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = smf.mixedlm(
                    formula,
                    work,
                    groups=work["mouse_cluster"],
                    re_formula="1",
                    vc_formula={"session": "0 + C(session_key)"},
                ).fit(reml=False, method=method, maxiter=1200, disp=False)
            fit = candidate
            optimizer = method
            if bool(getattr(candidate, "converged", False)):
                break
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    if fit is None:
        return pd.DataFrame(), "Model could not be fitted: " + " | ".join(errors)
    term = [name for name in fit.fe_params.index if "compartment" in name][0]
    estimate = float(fit.fe_params[term])
    se = float(fit.bse_fe[term])
    z_value = estimate / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    mouse_variance = (
        float(np.asarray(fit.cov_re, dtype=float).ravel()[0])
        if np.asarray(fit.cov_re).size else np.nan
    )
    session_variance = (
        float(np.asarray(fit.vcomp, dtype=float).ravel()[0])
        if np.asarray(fit.vcomp).size else np.nan
    )
    boundary = any(np.isfinite(value) and value <= 1e-8 for value in [mouse_variance, session_variance])
    result = pd.DataFrame([{
        "Comparison": "Pia minus Dura baseline diameter",
        "Estimate (um)": estimate,
        "SE": se,
        "z": z_value,
        "p-value": p_value,
        "Dura vessels": int(work["compartment"].eq("dura").sum()),
        "Pia vessels": int(work["compartment"].eq("pia").sum()),
        "Mouse clusters": work["mouse_cluster"].nunique(),
        "Sessions": work["session_key"].nunique(),
        "Optimizer": optimizer,
        "Mouse-intercept variance": mouse_variance,
        "Session-intercept variance": session_variance,
        "Boundary/singular": boundary,
        "Converged": bool(getattr(fit, "converged", False)),
        "Software": f"statsmodels {statsmodels.__version__}; Python {platform.python_version()}",
    }])
    return result, (
        "Maximum-likelihood vessel-level linear mixed-effects model adjusted for paper study/source: "
        "baseline diameter ~ compartment + study/source + (1 | mouse) + (1 | session nested within mouse). "
        "The coefficient is Pia minus Dura. Every accepted vessel contributes once."
    )

def fit_paper_compartment_slope(
    frame: pd.DataFrame,
    compartment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fit PACAP38 Response 1 slope within one vascular compartment."""
    label = {"dura": "Dura", "pia": "Pia"}[compartment]
    work = frame.dropna(
        subset=["baseline_um", "z_response", "mouse_alias", "dataset_id"]
    ).copy()
    if (
        len(work) < 4
        or work["baseline_um"].nunique() < 2
        or work["mouse_alias"].nunique() < 2
    ):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            "At least four vessels, two baseline diameters, and two mice are required.",
        )
    ols = stats.linregress(work["baseline_um"], work["z_response"])
    spearman = stats.spearmanr(work["baseline_um"], work["z_response"])
    center = float(work["baseline_um"].mean())
    work["baseline_centered_um"] = work["baseline_um"] - center
    formula = "z_response ~ baseline_centered_um"
    fit, optimizer, errors = None, "", []
    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = smf.mixedlm(
                    formula, work, groups=work["mouse_alias"].astype(str)
                ).fit(reml=False, method=method, maxiter=1000, disp=False)
            fit, optimizer = candidate, method
            if bool(candidate.converged):
                break
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    if fit is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            "Mixed-effects slope could not be estimated. " + " | ".join(errors),
        )
    try:
        slope = float(fit.fe_params["baseline_centered_um"])
        se = float(fit.bse_fe["baseline_centered_um"])
        z_value = slope / se if se > 0 else np.nan
        p_value = (
            float(2 * stats.norm.sf(abs(z_value)))
            if np.isfinite(z_value)
            else np.nan
        )
        random_variance = (
            float(np.asarray(fit.cov_re).ravel()[0])
            if np.asarray(fit.cov_re).size
            else 0.0
        )
        residual_variance = float(fit.scale)
        fixed_prediction = np.asarray(
            fit.model.exog @ fit.fe_params.to_numpy(float), dtype=float
        )
        fixed_variance = float(np.var(fixed_prediction, ddof=0))
        total_variance = (
            fixed_variance
            + max(random_variance, 0.0)
            + max(residual_variance, 0.0)
        )
        marginal_r2 = (
            fixed_variance / total_variance if total_variance > 0 else np.nan
        )
        conditional_r2 = (
            (fixed_variance + max(random_variance, 0.0)) / total_variance
            if total_variance > 0
            else np.nan
        )
        result = pd.DataFrame(
            [
                {
                    "Compartment": label,
                    "Slope (Z-score per um)": slope,
                    "SE": se,
                    "z": z_value,
                    "Raw p": p_value,
                    "OLS R-squared (descriptive)": float(ols.rvalue**2),
                    "Spearman rho (descriptive)": float(spearman.statistic),
                    "Spearman p (descriptive)": float(spearman.pvalue),
                    "Marginal R-squared (diameter)": marginal_r2,
                    "Conditional R-squared (diameter + mouse)": conditional_r2,
                    "Vessels": len(work),
                    "Mice": work["mouse_alias"].nunique(),
                    "Sessions": work["dataset_id"].nunique(),
                    "Converged": bool(fit.converged),
                }
            ]
        )
        diagnostics = pd.DataFrame(
            [
                {
                    "Compartment": label,
                    "Formula": "Response 1 Z-score ~ baseline diameter + (1 | mouse)",
                    "Software": (
                        f"statsmodels {statsmodels.__version__}; "
                        f"Python {platform.python_version()}"
                    ),
                    "Vessels": len(work),
                    "Unique mice": work["mouse_alias"].nunique(),
                    "Unique sessions": work["dataset_id"].nunique(),
                    "Baseline centering value (um)": center,
                    "Optimizer": optimizer,
                    "Converged": bool(fit.converged),
                    "Random-intercept variance": random_variance,
                    "Residual variance": residual_variance,
                    "Log likelihood": float(fit.llf),
                }
            ]
        )
        note = (
            "Maximum-likelihood linear mixed-effects model: Response 1 Z-score ~ "
            "baseline diameter + (1 | mouse). The slope is the expected Z-score "
            "difference per 1 um larger baseline diameter. OLS R-squared, the straight "
            "trend line, and Spearman rho are descriptive because they do not account "
            "for vessels clustered within mouse."
        )
        return result, diagnostics, note
    except Exception as exc:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"The model fit was returned, but inference could not be calculated: {exc}",
        )
