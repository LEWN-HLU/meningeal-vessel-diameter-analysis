# Vessel imaging analysis methods.
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
from ravi.paper2 import classify_vessel_sizes, _load_compartment_labels_for_payload

def read_vessel_trace(trace_path: str) -> pd.DataFrame:
    return pd.read_csv(trace_path)

def accepted_vessel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    required = {"Vessel ID", "Time (s)", "Diameter (um)"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    data["Time (s)"] = pd.to_numeric(data["Time (s)"], errors="coerce")
    data["Diameter (um)"] = pd.to_numeric(data["Diameter (um)"], errors="coerce")
    data = data.dropna(subset=["Time (s)", "Diameter (um)"])
    if "Vessel Status" in data.columns:
        data = data[
            data["Vessel Status"].astype(str).str.lower().str.strip().eq("accepted")
        ].copy()
    data["Vessel ID"] = data["Vessel ID"].astype(str).str.strip()
    return data

def trace_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    pivot = (
        data.groupby(["Time (s)", "Vessel ID"], observed=True)["Diameter (um)"]
        .mean()
        .unstack("Vessel ID")
        .sort_index()
    )
    times_s = pivot.index.to_numpy(dtype=float)
    vessel_ids = [str(column) for column in pivot.columns]
    matrix = pivot.to_numpy(dtype=float)
    return times_s, vessel_ids, matrix

def compute_z_matrix(
    matrix: np.ndarray,
    times_s: np.ndarray,
    baseline_start_min: float,
    baseline_end_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline_mask = (
        (times_s >= float(baseline_start_min) * 60.0)
        & (times_s < float(baseline_end_min) * 60.0)
    )
    if not baseline_mask.any():
        baseline_mask = times_s < float(baseline_end_min) * 60.0
    baseline = matrix[baseline_mask, :]
    baseline_mean = np.nanmean(baseline, axis=0)
    baseline_sd = np.nanstd(baseline, axis=0)
    baseline_sd[~np.isfinite(baseline_sd) | (baseline_sd == 0)] = np.nan
    z = (matrix - baseline_mean) / baseline_sd
    return z, baseline_mean, baseline_sd, baseline_mask

def dose_window_values(
    times_s: np.ndarray,
    z_matrix: np.ndarray,
    metadata: pd.DataFrame,
    dose_table: pd.DataFrame,
    response_min: float,
    value_col: str = "mean_z_response",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dose_order, dose in dose_table.reset_index(drop=True).iterrows():
        label = str(dose.get("Dose label", f"Dose {dose_order + 1}")).strip() or f"Dose {dose_order + 1}"
        start_min = float(dose.get("Recorded video time min", dose.get("Time min")))
        end_min = start_min + float(response_min)
        mask = (times_s >= start_min * 60.0) & (times_s < end_min * 60.0)
        if not mask.any():
            continue
        values = np.nanmean(z_matrix[mask, :], axis=0)
        for vessel_index, value in enumerate(values):
            rows.append(
                {
                    "dose_order": int(dose_order + 1),
                    "dose_label": label,
                    "dose_recorded_time_min": start_min,
                    "dose_admin_time_min": float(dose.get("Admin time min", start_min)),
                    "window_start_min": start_min,
                    "window_end_min": end_min,
                    "window_duration_min": float(response_min),
                    "window_recorded_frames": int(mask.sum()),
                    "vessel_index": int(vessel_index),
                    "vessel_id": metadata.iloc[vessel_index]["vessel_id"],
                    value_col: float(value) if np.isfinite(value) else np.nan,
                    "baseline_um": metadata.iloc[vessel_index]["baseline_um"],
                    "vessel_size": metadata.iloc[vessel_index]["vessel_size"],
                    "compartment": metadata.iloc[vessel_index]["compartment"],
                }
            )
    return pd.DataFrame(rows)

def dose_full_period_values(
    times_s: np.ndarray,
    z_matrix: np.ndarray,
    metadata: pd.DataFrame,
    dose_table: pd.DataFrame,
    value_col: str = "mean_z_response",
) -> pd.DataFrame:
    """Use recorded frames from each dose until the next dose or recording end."""
    if times_s.size == 0 or dose_table.empty:
        return pd.DataFrame()
    finite_times = times_s[np.isfinite(times_s)]
    if finite_times.size == 0:
        return pd.DataFrame()
    recording_end_s = float(finite_times.max())
    ordered_doses = dose_table.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for dose_index, dose in ordered_doses.iterrows():
        start_min = float(dose.get("Recorded video time min", dose.get("Time min")))
        if not np.isfinite(start_min):
            continue
        next_start_min = (
            float(ordered_doses.iloc[dose_index + 1]["Recorded video time min"])
            if dose_index + 1 < len(ordered_doses)
            else np.inf
        )
        end_s = min(next_start_min * 60.0, recording_end_s)
        start_s = start_min * 60.0
        if not np.isfinite(end_s) or end_s <= start_s:
            continue
        if next_start_min * 60.0 <= recording_end_s:
            mask = (times_s >= start_s) & (times_s < end_s)
        else:
            mask = (times_s >= start_s) & (times_s <= end_s)
        if not mask.any():
            continue
        values = np.nanmean(z_matrix[mask, :], axis=0)
        label = str(dose.get("Dose label", f"Dose {dose_index + 1}")).strip()
        for vessel_index, value in enumerate(values):
            rows.append(
                {
                    "dose_order": int(dose_index + 1),
                    "dose_label": label or f"Dose {dose_index + 1}",
                    "dose_recorded_time_min": start_min,
                    "dose_admin_time_min": float(dose.get("Admin time min", start_min)),
                    "window_start_min": start_min,
                    "window_end_min": end_s / 60.0,
                    "window_duration_min": end_s / 60.0 - start_min,
                    "window_recorded_frames": int(mask.sum()),
                    "vessel_index": int(vessel_index),
                    "vessel_id": metadata.iloc[vessel_index]["vessel_id"],
                    value_col: float(value) if np.isfinite(value) else np.nan,
                    "baseline_um": metadata.iloc[vessel_index]["baseline_um"],
                    "vessel_size": metadata.iloc[vessel_index]["vessel_size"],
                    "compartment": metadata.iloc[vessel_index]["compartment"],
                }
            )
    return pd.DataFrame(rows)

def baseline_z_values(
    z_matrix: np.ndarray,
    metadata: pd.DataFrame,
    baseline_mask: np.ndarray,
    baseline_start_min: float,
    first_recorded_min: float,
    value_col: str = "mean_z_response",
) -> pd.DataFrame:
    """Make Dose 0 from the recorded pre-first-dose frames for each vessel."""
    if not baseline_mask.any() or z_matrix.shape[1] == 0:
        return pd.DataFrame()
    values = np.nanmean(z_matrix[baseline_mask, :], axis=0)
    rows = []
    for vessel_index, value in enumerate(values):
        # Avoid floating-point residue such as -2e-16 in the Prism reference column.
        if value_col == "mean_z_response" and np.isfinite(value) and np.isclose(
            value, 0.0, atol=1e-12, rtol=0
        ):
            value = 0.0
        rows.append(
            {
                "dose_order": 0,
                "dose_label": "Baseline",
                "dose_recorded_time_min": baseline_start_min,
                "dose_admin_time_min": np.nan,
                "window_start_min": baseline_start_min,
                "window_end_min": first_recorded_min,
                "window_duration_min": first_recorded_min - baseline_start_min,
                "window_recorded_frames": int(baseline_mask.sum()),
                "vessel_index": int(vessel_index),
                "vessel_id": metadata.iloc[vessel_index]["vessel_id"],
                value_col: float(value) if np.isfinite(value) else np.nan,
                "baseline_um": metadata.iloc[vessel_index]["baseline_um"],
                "vessel_size": metadata.iloc[vessel_index]["vessel_size"],
                "compartment": metadata.iloc[vessel_index]["compartment"],
            }
        )
    return pd.DataFrame(rows)

def animal_level_dose_summary(values: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Collapse vessel rows to one animal/session mean per dose, optionally within category."""
    if values.empty:
        return pd.DataFrame()
    working = values.copy()
    if group_col is None:
        working["category"] = "All vessels"
    else:
        working["category"] = working[group_col].astype(str)
    grouped = (
        working.groupby(["category", "dose_order", "dose_label", "dose_recorded_time_min", "dose_admin_time_min"], observed=True, dropna=False)["mean_z_response"]
        .agg(
            animal_session_mean_z="mean",
            vessel_sd="std",
            n_vessels="count",
        )
        .reset_index()
        .sort_values(["category", "dose_order"])
    )
    grouped["vessel_sem_descriptive"] = grouped["vessel_sd"] / np.sqrt(grouped["n_vessels"])
    return grouped

def analyze_manifest_session_doses(
    manifest_row: pd.Series,
    timing_table: pd.DataFrame,
    baseline_duration_min: float,
    response_min: float,
) -> tuple[pd.DataFrame, ...]:
    """Return Z-score responses plus fixed/full raw-diameter vessel values."""
    trace_path = str(manifest_row["trace_path"])
    raw = read_vessel_trace(trace_path)
    vessel_data = accepted_vessel_frame(raw)
    times_s, vessel_ids, raw_matrix = trace_matrix(vessel_data)
    if raw_matrix.size == 0 or not vessel_ids or timing_table.empty:
        return (pd.DataFrame(),) * 8

    first_recorded_min = float(timing_table["Recorded video time min"].min())
    baseline_start_min = max(0.0, first_recorded_min - float(baseline_duration_min))
    z_matrix, baseline_um, baseline_sd, baseline_mask = compute_z_matrix(
        raw_matrix,
        times_s,
        baseline_start_min,
        first_recorded_min,
    )
    finite_vessels = np.isfinite(baseline_um) & np.isfinite(baseline_sd)
    raw_finite_vessels = np.isfinite(baseline_um)
    if not raw_finite_vessels.any():
        return (pd.DataFrame(),) * 8

    payload = {"trace_path": trace_path, "vessel_data": vessel_data}
    compartments, compartment_rgbs, _, _ = _load_compartment_labels_for_payload(
        payload,
        vessel_ids,
    )
    all_metadata = pd.DataFrame(
        {
            "vessel_index": np.arange(len(vessel_ids)),
            "vessel_id": vessel_ids,
            "baseline_um": baseline_um,
            "baseline_sd_um": baseline_sd,
            "vessel_size": classify_vessel_sizes(baseline_um),
            "compartment": compartments,
            "compartment_rgb": compartment_rgbs,
            "finite_baseline": finite_vessels,
        }
    )
    raw_metadata = all_metadata.loc[raw_finite_vessels].reset_index(drop=True)
    raw_keep_idx = raw_metadata["vessel_index"].astype(int).to_numpy()
    raw_kept_matrix = raw_matrix[:, raw_keep_idx]
    raw_metadata["vessel_index"] = np.arange(len(raw_metadata))
    metadata = all_metadata[all_metadata["finite_baseline"]].reset_index(drop=True)
    keep_idx = metadata["vessel_index"].astype(int).to_numpy()
    z_matrix = z_matrix[:, keep_idx]
    metadata["vessel_index"] = np.arange(len(metadata))

    animal_session = str(manifest_row.get("source_folder", manifest_row.get("mouse_alias", "")))
    identifiers = {
        "animal_session": animal_session,
        "mouse_alias": str(manifest_row.get("mouse_alias", "")),
        "study": str(manifest_row.get("study_label", "")),
        "trace_path": trace_path,
    }
    def summarize_vessels(vessel_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if vessel_values.empty:
            return pd.DataFrame(), pd.DataFrame()
        for column, value in identifiers.items():
            vessel_values[column] = value
        summaries = [animal_level_dose_summary(vessel_values)]
        summaries.append(
            animal_level_dose_summary(
                vessel_values[
                    vessel_values["vessel_size"].astype(str).isin(["Small", "Large"])
                ],
                "vessel_size",
            )
        )
        summaries.append(
            animal_level_dose_summary(
                vessel_values[
                    vessel_values["compartment"].astype(str).isin(["dura", "pia"])
                ],
                "compartment",
            )
        )
        animal_summary = pd.concat(
            [frame for frame in summaries if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        for column, value in identifiers.items():
            animal_summary[column] = value
        return animal_summary, vessel_values

    def summarize_window(window_min: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        vessel_values = (
            dose_full_period_values(times_s, z_matrix, metadata, timing_table)
            if window_min is None
            else dose_window_values(
                times_s, z_matrix, metadata, timing_table, window_min
            )
        )
        return summarize_vessels(vessel_values)

    adjustable_summary, adjustable_vessels = summarize_window(response_min)
    if np.isclose(response_min, 10.0):
        fixed_summary, fixed_vessels = (
            adjustable_summary.copy(),
            adjustable_vessels.copy(),
        )
    else:
        fixed_summary, fixed_vessels = summarize_window(10.0)
    full_summary, full_vessels = summarize_window(None)
    baseline_summary, baseline_vessels = summarize_vessels(
        baseline_z_values(
            z_matrix, metadata, baseline_mask, baseline_start_min, first_recorded_min
        )
    )
    if not baseline_summary.empty:
        fixed_summary = pd.concat(
            [baseline_summary, fixed_summary], ignore_index=True, sort=False
        )
        fixed_vessels = pd.concat(
            [baseline_vessels, fixed_vessels], ignore_index=True, sort=False
        )
        full_summary = pd.concat(
            [baseline_summary, full_summary], ignore_index=True, sort=False
        )
        full_vessels = pd.concat(
            [baseline_vessels, full_vessels], ignore_index=True, sort=False
        )
    if not full_summary.empty:
        window_details = full_vessels[
            [
                "dose_order", "window_start_min", "window_end_min",
                "window_duration_min", "window_recorded_frames",
            ]
        ].drop_duplicates("dose_order")
        full_summary = full_summary.merge(
            window_details, on="dose_order", how="left", validate="many_to_one"
        )

    raw_baseline = baseline_z_values(
        raw_kept_matrix, raw_metadata, baseline_mask,
        baseline_start_min, first_recorded_min,
        value_col="mean_diameter_um",
    )
    raw_fixed = dose_window_values(
        times_s, raw_kept_matrix, raw_metadata, timing_table, 10.0,
        value_col="mean_diameter_um",
    )
    raw_full = dose_full_period_values(
        times_s, raw_kept_matrix, raw_metadata, timing_table,
        value_col="mean_diameter_um",
    )
    def prepare_raw_values(postdose: pd.DataFrame) -> pd.DataFrame:
        values = pd.concat([raw_baseline, postdose], ignore_index=True, sort=False)
        if values.empty:
            return values
        values["change_from_baseline_um"] = (
            values["mean_diameter_um"] - values["baseline_um"]
        )
        values.loc[values["dose_order"].eq(0), "change_from_baseline_um"] = 0.0
        for column, value in identifiers.items():
            values[column] = value
        return values

    raw_fixed = prepare_raw_values(raw_fixed)
    raw_full = prepare_raw_values(raw_full)
    return (
        adjustable_summary, adjustable_vessels, fixed_summary, fixed_vessels,
        full_summary, full_vessels, raw_fixed, raw_full,
    )

def summarize_dose_across_mice(
    session_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average repeated sessions within mouse before calculating dose mean and SEM."""
    if session_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = session_summary.copy()
    work["animal_session_mean_z"] = pd.to_numeric(
        work["animal_session_mean_z"], errors="coerce"
    )
    work["mouse_alias"] = work["mouse_alias"].astype(str).str.strip()
    work = work[
        work["animal_session_mean_z"].notna()
        & ~work["mouse_alias"].str.lower().isin(["", "nan", "none", "<na>"])
    ]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    mouse_values = (
        work.groupby(
            ["study", "mouse_alias", "category", "dose_order", "dose_label"],
            observed=True,
            as_index=False,
        )
        .agg(
            mouse_mean_z=("animal_session_mean_z", "mean"),
            imaging_sessions=("trace_path", "nunique"),
            accepted_vessel_contributions=("n_vessels", "sum"),
        )
    )
    group_summary = (
        mouse_values.groupby(
            ["study", "category", "dose_order", "dose_label"],
            observed=True,
            as_index=False,
        )
        .agg(
            mean_z_across_mice=("mouse_mean_z", "mean"),
            sd_across_mice=("mouse_mean_z", "std"),
            n_mice=("mouse_alias", "nunique"),
            n_imaging_sessions=("imaging_sessions", "sum"),
            accepted_vessel_contributions=("accepted_vessel_contributions", "sum"),
        )
    )
    group_summary["sem_across_mice"] = (
        group_summary["sd_across_mice"] / np.sqrt(group_summary["n_mice"])
    )
    return mouse_values, group_summary

def prism_mouse_wide(
    mouse_values: pd.DataFrame, value_col: str = "mouse_mean_z"
) -> pd.DataFrame:
    """Place repeated dose values from each mouse in adjacent Prism columns."""
    if mouse_values.empty:
        return pd.DataFrame()
    values = mouse_values.copy()
    values["Dose"] = (
        "Dose " + values["dose_order"].astype(str)
        + ": " + values["dose_label"].astype(str)
    )
    wide = values.pivot_table(
        index=["study", "mouse_alias", "category"],
        columns="Dose",
        values=value_col,
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    ordered_doses = (
        values[["dose_order", "Dose"]]
        .drop_duplicates()
        .sort_values("dose_order")["Dose"]
        .tolist()
    )
    return wide[["study", "mouse_alias", "category"] + ordered_doses]
