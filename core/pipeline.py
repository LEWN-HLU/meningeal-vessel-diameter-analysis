# Vessel imaging analysis methods.
from __future__ import annotations
from copy import deepcopy
from typing import Any
import numpy as np
import pandas as pd
from ravi.paper2 import attach_dataset_metadata, build_vessel_qc_df, build_z_response_window_df, canonical_mouse_alias, discover_paper2_datasets, load_paper2_vessel_data, validate_paper2_vessel_data
from ravi.preprocessing import filter_accepted_vessels, segment_vessels

PACAP_ANTIPACAP_STUDY = "PACAP / Anti-PACAP"

PACAP_DOSING_STUDY = "PACAP / Dosing study"

PACAP_VEHICLE_GROUPS = {"veh", "vehicle"}

PACAP_VEHICLE_SUFFIX = "__pooled_dosing_vehicle"

def _pooled_dataset_id(dataset_id: str, occupied: set[str], suffix: str) -> str:
    candidate = f"{dataset_id}{suffix}"
    counter = 2
    while candidate in occupied:
        candidate = f"{dataset_id}{suffix}_{counter}"
        counter += 1
    occupied.add(candidate)
    return candidate

def pool_pacap_dosing_vehicle(
    vessels: dict,
    segmented: dict,
    response_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    source_groups: set[str] | None = None,
    target_group: str | None = None,
    suffix: str = PACAP_VEHICLE_SUFFIX,
    note: str = "PACAP dosing study vehicle pooled into PACAP/anti-PACAP study",
    remove_source_sessions: bool = False,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Copy dosing-study vehicle sessions into the PACAP/Anti-PACAP study."""
    pooled_vessels = deepcopy(vessels)
    pooled_segmented = deepcopy(segmented)
    mappings: list[dict[str, str]] = []
    allowed_groups = source_groups or PACAP_VEHICLE_GROUPS
    for group, datasets in vessels.items():
        if str(group).lower() not in allowed_groups:
            continue
        pooled_group = target_group or str(group)
        occupied = {str(value) for value in pooled_vessels.get(pooled_group, {})}
        for dataset_id, payload in datasets.items():
            study = str(payload.get("study_label", payload.get("study", "")))
            if study != PACAP_DOSING_STUDY:
                continue
            new_dataset_id = _pooled_dataset_id(str(dataset_id), occupied, suffix)
            mappings.append({
                "source_group": str(group),
                "group": pooled_group,
                "original_dataset_id": str(dataset_id),
                "pooled_dataset_id": new_dataset_id,
                "from_study": PACAP_DOSING_STUDY,
                "pooled_as_study": PACAP_ANTIPACAP_STUDY,
                "note": note,
            })
            copied_payload = deepcopy(payload)
            copied_payload.update({
                "original_study": study,
                "original_study_label": study,
                "original_dataset_id": str(dataset_id),
                "original_group": str(group),
                "original_treatment": str(group),
                "pooled_source_note": note,
                "study": PACAP_ANTIPACAP_STUDY,
                "study_label": PACAP_ANTIPACAP_STUDY,
            })
            pooled_vessels.setdefault(pooled_group, {})[new_dataset_id] = copied_payload
            segment_payload = segmented.get(group, {}).get(dataset_id)
            if segment_payload is not None:
                copied_segment = deepcopy(segment_payload)
                copied_segment.update({
                    "original_study": study,
                    "original_study_label": study,
                    "original_dataset_id": str(dataset_id),
                    "original_group": str(group),
                    "original_treatment": str(group),
                    "pooled_source_note": copied_payload["pooled_source_note"],
                    "study": PACAP_ANTIPACAP_STUDY,
                    "study_label": PACAP_ANTIPACAP_STUDY,
                })
                pooled_segmented.setdefault(pooled_group, {})[new_dataset_id] = copied_segment

    def append_table_copies(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not mappings:
            return frame.copy()
        copies: list[pd.DataFrame] = []
        for mapping in mappings:
            mask = (
                frame["group"].astype(str).eq(mapping["source_group"])
                & frame["dataset_id"].astype(str).eq(mapping["original_dataset_id"])
                & frame["study"].astype(str).eq(PACAP_DOSING_STUDY)
            )
            copied = frame.loc[mask].copy()
            if copied.empty:
                continue
            copied["original_study"] = copied["study"]
            copied["original_group"] = copied["group"]
            copied["original_treatment"] = copied["group"]
            copied["original_dataset_id"] = copied["dataset_id"].astype(str)
            copied["pooled_source_note"] = mapping["note"]
            copied["study"] = PACAP_ANTIPACAP_STUDY
            copied["group"] = mapping["group"]
            copied["dataset_id"] = mapping["pooled_dataset_id"]
            copies.append(copied)
        combined = (
            pd.concat([frame, *copies], ignore_index=True, sort=False)
            if copies
            else frame.copy()
        )
        if remove_source_sessions and mappings:
            source_keys = {
                (mapping["source_group"], mapping["original_dataset_id"])
                for mapping in mappings
            }
            is_source = combined.apply(
                lambda row: (
                    str(row["group"]), str(row["dataset_id"])
                ) in source_keys
                and str(row["study"]) == PACAP_DOSING_STUDY,
                axis=1,
            )
            combined = combined.loc[~is_source].copy()
        return combined

    if remove_source_sessions:
        for mapping in mappings:
            source_group = mapping["source_group"]
            source_dataset = mapping["original_dataset_id"]
            pooled_vessels.get(source_group, {}).pop(source_dataset, None)
            pooled_segmented.get(source_group, {}).pop(source_dataset, None)
        pooled_vessels = {
            group: datasets for group, datasets in pooled_vessels.items() if datasets
        }
        pooled_segmented = {
            group: datasets for group, datasets in pooled_segmented.items() if datasets
        }

    return (
        pooled_vessels,
        pooled_segmented,
        append_table_copies(response_df),
        append_table_copies(qc_df),
        pd.DataFrame(mappings),
    )

def prefer_dosing_pacap_for_overlapping_mice(
    vessels: dict,
    segmented: dict,
    response_df: pd.DataFrame,
    qc_df: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Keep dosing-study PACAP when its mouse also has an original PACAP session."""
    dosing_aliases = {
        canonical_mouse_alias(metrics.get("mouse_alias", ""))
        for metrics in segmented.get("PACAP", {}).values()
        if str(metrics.get("study_label", metrics.get("study", "")))
        == PACAP_ANTIPACAP_STUDY
        and str(metrics.get("original_study", metrics.get("original_study_label", "")))
        == PACAP_DOSING_STUDY
    }
    dosing_aliases.discard("")
    replacement_ids: list[str] = []
    audit_rows: list[dict[str, str]] = []
    excluded_segmented: dict[str, dict] = {}
    for dataset_id, metrics in list(segmented.get("PACAP", {}).items()):
        alias = canonical_mouse_alias(metrics.get("mouse_alias", ""))
        study = str(metrics.get("study_label", metrics.get("study", "")))
        original_study = str(
            metrics.get("original_study", metrics.get("original_study_label", ""))
        )
        if (
            alias in dosing_aliases
            and study == PACAP_ANTIPACAP_STUDY
            and original_study != PACAP_DOSING_STUDY
        ):
            replacement_ids.append(str(dataset_id))
            excluded_segmented[str(dataset_id)] = deepcopy(metrics)
            audit_rows.append(
                {
                    "mouse_alias": alias,
                    "excluded_group": "PACAP",
                    "excluded_dataset_id": str(dataset_id),
                    "excluded_source_study": PACAP_ANTIPACAP_STUDY,
                    "retained_source_study": PACAP_DOSING_STUDY,
                    "retained_treatment": "PACAP 0.1 mg/kg",
                    "reason": (
                        "Mouse was present in both PACAP sources; dosing-study "
                        "0.1 mg/kg session retained by the prespecified replacement rule."
                    ),
                }
            )
    if not replacement_ids:
        return (
            vessels,
            segmented,
            response_df,
            qc_df,
            pd.DataFrame(),
            {},
        )

    kept_vessels = deepcopy(vessels)
    kept_segmented = deepcopy(segmented)
    excluded_vessels = {
        dataset_id: deepcopy(kept_vessels.get("PACAP", {}).get(dataset_id, {}))
        for dataset_id in replacement_ids
    }
    for dataset_id in replacement_ids:
        kept_vessels.get("PACAP", {}).pop(dataset_id, None)
        kept_segmented.get("PACAP", {}).pop(dataset_id, None)

    def without_replaced(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if frame.empty:
            return frame.copy(), frame.copy()
        mask = (
            frame["group"].astype(str).eq("PACAP")
            & frame["dataset_id"].astype(str).isin(replacement_ids)
            & frame["study"].astype(str).eq(PACAP_ANTIPACAP_STUDY)
        )
        return frame.loc[~mask].copy(), frame.loc[mask].copy()

    kept_response, excluded_response = without_replaced(response_df)
    kept_qc, excluded_qc = without_replaced(qc_df)
    diagnostic = {
        "vessels": {"PACAP": excluded_vessels},
        "segmented": {"PACAP": excluded_segmented},
        "response_df": excluded_response,
        "qc_df": excluded_qc,
    }
    return (
        kept_vessels,
        kept_segmented,
        kept_response,
        kept_qc,
        pd.DataFrame(audit_rows),
        diagnostic,
    )

def scan_data(
    root: str,
    include_old: bool = False,
    allow_inscopix_without_acceleration: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    return discover_paper2_datasets(
        root,
        include_old=include_old,
        allow_inscopix_without_acceleration=allow_inscopix_without_acceleration,
    )

def load_selected(
    root: str,
    manifest: pd.DataFrame,
    studies: list[str],
    treatments: list[str],
) -> tuple[dict, pd.DataFrame, list[str]]:
    return load_paper2_vessel_data(
        root,
        manifest=manifest,
        study_labels=studies,
        treatments=treatments,
        preferred_only=True,
    )

def run_pipeline(
    vessels: dict,
    baseline_sec: int,
    response_sec: int,
    sampling_rate: int,
    gap_threshold_sec: float,
    max_response_windows: int | None,
    skip_invalid_datasets: bool,
    extreme_z_threshold: float | None = None,
    pool_pacap_vehicle: bool = False,
    pool_pacap_dose: bool = False,
) -> dict[str, Any]:
    """Validate, filter, segment, and create compact analysis tables."""
    issues = validate_paper2_vessel_data(
        vessels,
        gap_threshold_sec=gap_threshold_sec,
        accepted_only=True,
        min_response_windows=max_response_windows or 1,
    )
    analysis_vessels = deepcopy(vessels)
    skipped = pd.DataFrame()
    if not issues.empty:
        if not skip_invalid_datasets:
            return {"issues": issues, "ready": False}
        skipped = issues.copy()
        bad = set(zip(issues["Group"].astype(str), issues["Dataset ID"].astype(str)))
        analysis_vessels = {
            str(group): {
                str(dataset_id): payload
                for dataset_id, payload in datasets.items()
                if (str(group), str(dataset_id)) not in bad
            }
            for group, datasets in analysis_vessels.items()
        }
        analysis_vessels = {group: rows for group, rows in analysis_vessels.items() if rows}

    analysis_vessels = filter_accepted_vessels(analysis_vessels)
    extreme_excluded = pd.DataFrame()
    if extreme_z_threshold is not None and np.isfinite(float(extreme_z_threshold)):
        extreme_excluded = extreme_z_qc(
            analysis_vessels,
            baseline_frames=int(baseline_sec) * int(sampling_rate),
            threshold=float(extreme_z_threshold),
        )
        analysis_vessels = exclude_extreme_z_vessels(
            analysis_vessels,
            extreme_excluded,
        )
    segmented = segment_vessels(
        analysis_vessels,
        baseline_sec=int(baseline_sec),
        response_sec=int(response_sec),
        sampling_rate=int(sampling_rate),
        gap_threshold_sec=float(gap_threshold_sec),
        response_window_mode="all_fixed",
        max_response_windows=max_response_windows,
    )
    segmented = attach_dataset_metadata(segmented, analysis_vessels)
    response_df = build_z_response_window_df(segmented)
    qc_df, label_warnings = build_vessel_qc_df(segmented, analysis_vessels)
    response_df = merge_vessel_metadata(response_df, qc_df)
    pooling_manifest = pd.DataFrame()
    replacement_audit = pd.DataFrame()
    replacement_diagnostic: dict = {}
    if pool_pacap_vehicle:
        (
            analysis_vessels,
            segmented,
            response_df,
            qc_df,
            pooling_manifest,
        ) = pool_pacap_dosing_vehicle(
            analysis_vessels,
            segmented,
            response_df,
            qc_df,
            remove_source_sessions=True,
        )
    if pool_pacap_dose:
        dose_manifest = pd.DataFrame()
        (
            analysis_vessels,
            segmented,
            response_df,
            qc_df,
            dose_manifest,
        ) = pool_pacap_dosing_vehicle(
            analysis_vessels,
            segmented,
            response_df,
            qc_df,
            source_groups={"0_1mgkg_pacap"},
            target_group="PACAP",
            suffix="__pooled_0_1mgkg",
            note="0.1 mg/kg PACAP dosing pooled into PACAP/anti-PACAP study",
            remove_source_sessions=True,
        )
        (
            analysis_vessels,
            segmented,
            response_df,
            qc_df,
            replacement_audit,
            replacement_diagnostic,
        ) = prefer_dosing_pacap_for_overlapping_mice(
            analysis_vessels, segmented, response_df, qc_df
        )
        pooling_manifest = pd.concat(
            [pooling_manifest, dose_manifest],
            ignore_index=True,
            sort=False,
        )

    loaded = {
        (str(group), str(dataset_id))
        for group, datasets in analysis_vessels.items()
        for dataset_id in datasets
    }
    completed = {
        (str(group), str(dataset_id))
        for group, datasets in segmented.items()
        for dataset_id in datasets
    }
    segment_warnings = [
        f"Skipped during segmentation: {group} / {dataset_id}"
        for group, dataset_id in sorted(loaded - completed)
    ]
    return {
        "ready": True,
        "issues": issues,
        "skipped": skipped,
        "vessels": analysis_vessels,
        "segmented": segmented,
        "response_df": response_df,
        "qc_df": qc_df,
        "extreme_excluded": extreme_excluded,
        "label_warnings": label_warnings,
        "segment_warnings": segment_warnings,
        "n_datasets": len(completed),
        "pooling_manifest": pooling_manifest,
        "pacap_replacement_audit": replacement_audit,
        "pacap_replacement_diagnostic": replacement_diagnostic,
    }

def merge_vessel_metadata(response_df: pd.DataFrame, qc_df: pd.DataFrame) -> pd.DataFrame:
    if response_df.empty or qc_df.empty:
        return response_df.copy()
    keys = ["group", "dataset_id", "vessel_index"]
    metadata = qc_df[
        keys + ["vessel_id", "vessel_size", "compartment", "rgb"]
    ].drop_duplicates(keys)
    out = response_df.merge(metadata, on=keys, how="left")
    out["compartment"] = out["compartment"].fillna("unknown").astype(str).str.lower()
    out["vessel_size"] = out["vessel_size"].fillna("Unknown").astype(str)
    return out

def exclude_extreme_z_vessels(vessels: dict, flags: pd.DataFrame) -> dict:
    """Remove flagged Vessel IDs and drop sessions left without any retained vessels."""
    if flags.empty:
        return vessels
    flagged: dict[tuple[str, str], set[str]] = {}
    for _, row in flags.iterrows():
        key = (str(row["Treatment"]), str(row["Dataset"]))
        flagged.setdefault(key, set()).add(str(row["Vessel ID"]))

    filtered: dict = {}
    for group, datasets in vessels.items():
        retained_datasets = {}
        for dataset_id, payload in datasets.items():
            frame = payload.get("vessel_data", pd.DataFrame())
            bad_ids = flagged.get((str(group), str(dataset_id)), set())
            if bad_ids and "Vessel ID" in frame.columns:
                frame = frame[
                    ~frame["Vessel ID"].astype(str).isin(bad_ids)
                ].copy()
            if frame.empty:
                continue
            payload["vessel_data"] = frame
            retained_datasets[dataset_id] = payload
        if retained_datasets:
            filtered[group] = retained_datasets
    return filtered

def accepted_vessel_counts(vessels: dict) -> pd.DataFrame:
    rows = []
    for group, datasets in vessels.items():
        for dataset_id, payload in datasets.items():
            frame = payload.get("vessel_data", pd.DataFrame())
            accepted = frame
            if "Vessel Status" in frame.columns:
                accepted = frame[frame["Vessel Status"].astype(str).str.lower() == "accepted"]
            rows.append(
                {
                    "Study": payload.get("study_label", ""),
                    "Treatment": group,
                    "Dataset": dataset_id,
                    "Mouse alias": payload.get("mouse_alias", ""),
                    "Accepted vessels": int(accepted["Vessel ID"].nunique())
                    if "Vessel ID" in accepted.columns
                    else 0,
                    "Trace frames per vessel": int(accepted.groupby("Vessel ID").size().median())
                    if not accepted.empty and "Vessel ID" in accepted.columns
                    else 0,
                }
            )
    return pd.DataFrame(rows)

def extreme_z_qc(vessels: dict, baseline_frames: int, threshold: float) -> pd.DataFrame:
    rows = []
    for group, datasets in vessels.items():
        for dataset_id, payload in datasets.items():
            frame = payload.get("vessel_data", pd.DataFrame())
            if frame.empty or not {"Vessel ID", "Diameter (um)"}.issubset(frame.columns):
                continue
            if "Vessel Status" in frame.columns:
                frame = frame[frame["Vessel Status"].astype(str).str.lower() == "accepted"]
            for vessel_id, vessel in frame.groupby("Vessel ID", sort=False):
                vessel = vessel.sort_values("Time (s)")
                diameter = pd.to_numeric(vessel["Diameter (um)"], errors="coerce").to_numpy(float)
                baseline = diameter[: min(int(baseline_frames), len(diameter))]
                mean = np.nanmean(baseline)
                sd = np.nanstd(baseline, ddof=0)
                max_abs = np.nan
                if np.isfinite(sd) and sd > 0:
                    max_abs = float(np.nanmax(np.abs((diameter - mean) / sd)))
                if np.isfinite(max_abs) and max_abs >= threshold:
                    rows.append(
                        {
                            "Study": payload.get("study_label", ""),
                            "Treatment": group,
                            "Dataset": dataset_id,
                            "Mouse alias": payload.get("mouse_alias", ""),
                            "Vessel ID": vessel_id,
                            "Maximum absolute z-score": max_abs,
                            "Flag threshold": threshold,
                        }
                    )
    return pd.DataFrame(rows)
