# Vessel imaging analysis methods.
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from scipy import stats

Z_THRESHOLD = 1.96

def build_vessel_values(
    response_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    period_choice: str,
) -> pd.DataFrame:
    keys = ["group", "dataset_id", "vessel_index"]
    metadata_columns = keys + [
        column
        for column in ["study", "mouse_alias", "vessel_id", "baseline_um", "vessel_size", "compartment"]
        if column in qc_df.columns
    ]
    metadata = qc_df[metadata_columns].drop_duplicates(keys)

    work = response_df.copy()
    if period_choice == "Response 1 only":
        work = work[work["window_order"].astype(int) == 1]

    value_keys = ["study", "group", "dataset_id", "mouse_alias", "vessel_index"]
    values = (
        work.groupby(value_keys, observed=True, as_index=False)
        .agg(
            z_response=("z_response", "mean"),
            response_periods=("window_order", "nunique"),
        )
    )
    merge_metadata = metadata.drop(
        columns=[column for column in ["study", "mouse_alias"] if column in metadata.columns]
    )
    values = values.merge(merge_metadata, on=keys, how="inner", validate="many_to_one")
    values = values[np.isfinite(values["z_response"]) & np.isfinite(values["baseline_um"])].copy()
    values["response_class"] = "Neutral response"
    values.loc[values["z_response"] >= Z_THRESHOLD, "response_class"] = "Positive response"
    values.loc[values["z_response"] <= -Z_THRESHOLD, "response_class"] = "Negative response"
    values["agent_key"] = values["study"].astype(str) + " | " + values["group"].astype(str)
    return values

def linear_relationship_stats(frame: pd.DataFrame) -> tuple[dict, str]:
    """Fit baseline-diameter slope while accounting for vessels clustered within mouse."""
    work = frame.dropna(subset=["baseline_um", "z_response", "mouse_alias"]).copy()
    empty = {
        "Vessels": len(work),
        "Mice": work["mouse_alias"].nunique() if not work.empty else 0,
        "Sessions": work["dataset_id"].nunique() if not work.empty else 0,
        "Slope (Z-score per um)": np.nan,
        "SE": np.nan,
        "z": np.nan,
        "Raw p": np.nan,
        "OLS R-squared (descriptive)": np.nan,
        "Spearman rho (descriptive)": np.nan,
        "Marginal R-squared (diameter)": np.nan,
        "Conditional R-squared (diameter + mouse)": np.nan,
        "Converged": False,
        "Random-intercept variance": np.nan,
        "Residual variance": np.nan,
        "Software": f"statsmodels {statsmodels.__version__}",
    }
    if len(work) < 4 or work["baseline_um"].nunique() < 2 or work["mouse_alias"].nunique() < 2:
        return empty, "At least four vessels, two distinct baseline diameters, and two mice are required."
    ols = stats.linregress(work["baseline_um"], work["z_response"])
    spearman = stats.spearmanr(work["baseline_um"], work["z_response"])
    empty["OLS R-squared (descriptive)"] = float(ols.rvalue ** 2)
    empty["Spearman rho (descriptive)"] = float(spearman.statistic)
    center = float(work["baseline_um"].mean())
    work["baseline_centered_um"] = work["baseline_um"] - center
    fit, optimizer, errors = None, "", []
    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = smf.mixedlm(
                    "z_response ~ baseline_centered_um",
                    work,
                    groups=work["mouse_alias"].astype(str),
                ).fit(reml=False, method=method, maxiter=1000, disp=False)
            fit, optimizer = candidate, method
            if bool(candidate.converged):
                break
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    if fit is None:
        return empty, "Mixed-effects slope could not be estimated. " + " | ".join(errors)
    slope = float(fit.fe_params["baseline_centered_um"])
    se = float(fit.bse_fe["baseline_centered_um"])
    z_value = slope / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    random_variance = float(np.asarray(fit.cov_re).ravel()[0]) if np.asarray(fit.cov_re).size else 0.0
    residual_variance = float(fit.scale)
    fixed_prediction = np.asarray(fit.model.exog @ fit.fe_params.to_numpy(float), dtype=float)
    fixed_variance = float(np.var(fixed_prediction, ddof=0))
    total_variance = fixed_variance + max(random_variance, 0.0) + max(residual_variance, 0.0)
    marginal_r2 = fixed_variance / total_variance if total_variance > 0 else np.nan
    conditional_r2 = (fixed_variance + max(random_variance, 0.0)) / total_variance if total_variance > 0 else np.nan
    empty.update({
        "Slope (Z-score per um)": slope,
        "SE": se,
        "z": z_value,
        "Raw p": p_value,
        "Marginal R-squared (diameter)": marginal_r2,
        "Conditional R-squared (diameter + mouse)": conditional_r2,
        "Converged": bool(fit.converged),
        "Random-intercept variance": random_variance,
        "Residual variance": residual_variance,
        "Optimizer": optimizer,
        "Baseline centering value (um)": center,
    })
    note = (
        "Maximum-likelihood linear mixed-effects model: Response Z-score ~ baseline diameter + "
        "(1 | mouse). Marginal R-squared is variance explained by baseline diameter; conditional "
        "R-squared also includes between-mouse variance. OLS R-squared is shown only as the "
        "descriptive straight-line fit to the plotted vessel points."
    )
    return empty, note
