# Vessel imaging analysis methods.
from __future__ import annotations
import os
import re
from typing import Any
import numpy as np
import pandas as pd

TRACE_RE = re.compile(r"^Ves{2,3}el traces_(?P<meta>.+)\.csv$", re.IGNORECASE)

ACCEL_RE = re.compile(r"^Acceleration_(?P<meta>.+)\.csv$", re.IGNORECASE)

REQUIRED_VESSEL_COLUMNS = ("Vessel ID", "Time (s)", "Diameter (um)")

SIZE_SMALL_MAX_UM = 20.0

SIZE_LARGE_MIN_UM = 50.0

SIZE_MIDDLE_LABEL = "Excluded 20-50 um"

PACAP_DOSING_STUDY_LABEL = "PACAP / Dosing study"

PACAP_DOSING_PLAIN_TREATMENT = "0_1mgkg_PACAP"

def canonical_treatment(raw: str | None) -> str:
    """Normalize common treatment spellings without consulting metadata files."""
    if raw is None:
        return ""

    cleaned = str(raw).strip().strip("_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    key = cleaned.lower()

    # Some caffeine trace filenames include an animal/study prefix in the
    # treatment field (a study-prefixed caffeine token). The final underscore-
    # separated token is the actual treatment for these datasets.
    if key.split("_")[-1] == "caffeine":
        return "Caffeine"

    aliases = {
        "vehicle": "Veh",
        "veh": "Veh",
        "lev": "Levcro",
        "levcro": "Levcro",
        "levcromakalim": "Levcro",
        "lname": "LNAME",
        "ll": "LL",
        "pacap": "PACAP",
        "anti-pacap_mab": "Anti-PACAP mAb",
        "gr1": "GR1",
        "gr2": "GR2",
        "ntg": "Nitroglycerin",
        "nitroglycerin": "Nitroglycerin",
    }
    return aliases.get(key, cleaned)

def canonical_mouse_alias(value: Any) -> str:
    """Return a stable filename/display token for a mouse alias."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return str(int(number)) if number.is_integer() else text

def parse_file_metadata(filename: str) -> dict[str, Any] | None:
    """Parse mouse alias and treatment from a trace or acceleration filename."""
    match = TRACE_RE.match(filename) or ACCEL_RE.match(filename)
    if not match:
        return None

    meta = match.group("meta").strip()
    tokens = [token for token in meta.split("_") if token]
    if not tokens:
        return None

    mouse_alias: int | str | None
    mouse_alias = int(tokens[0]) if tokens[0].isdigit() else None
    treatment_raw = "_".join(tokens[1:]) if mouse_alias is not None else meta

    return {
        "filename_metadata": meta,
        "mouse_alias": mouse_alias,
        "treatment_raw": treatment_raw,
        "treatment": canonical_treatment(treatment_raw),
    }

def parse_folder_animal_id(folder_name: str, mouse_alias: int | str | None, treatment_raw: str) -> str:
    """Extract the non-treatment animal identifier from a source folder name."""
    if mouse_alias is None:
        return ""

    tokens = [token for token in re.split(r"_+", folder_name) if token]
    if not tokens or str(tokens[0]) != str(mouse_alias):
        return ""

    rest = tokens[1:]
    treatment_tokens = [token for token in str(treatment_raw).split("_") if token]
    if not rest:
        return ""

    if treatment_tokens:
        lowered_rest = [token.lower() for token in rest]
        lowered_treat = [token.lower() for token in treatment_tokens]
        n_treat = len(lowered_treat)

        for i in range(0, len(lowered_rest) - n_treat + 1):
            if lowered_rest[i : i + n_treat] == lowered_treat:
                return "_".join(rest[:i])

    last_token = canonical_treatment(rest[-1]).lower()
    if last_token == canonical_treatment(treatment_raw).lower():
        return "_".join(rest[:-1])

    return "_".join(rest)

def _study_label(relative_path: str) -> tuple[str, str]:
    parts = relative_path.split(os.sep)
    study = parts[0] if parts else ""
    substudy = ""
    if len(parts) > 1 and study.lower() == "pacap":
        substudy = parts[1]
    label = f"{study} / {substudy}" if substudy else study
    return study, label

def paper2_contextual_treatment(treatment_raw: str | None, study_label: str) -> str:
    """Normalize treatment names with study-specific Paper 2 context."""
    treatment = canonical_treatment(treatment_raw)
    study_key = str(study_label).lower()
    if "ntg" in study_key or "nitroglycerin" in study_key:
        if treatment == "GR1":
            return "Veh"
        if treatment == "GR2":
            return "Nitroglycerin"
    if study_label == PACAP_DOSING_STUDY_LABEL and treatment == "PACAP":
        return PACAP_DOSING_PLAIN_TREATMENT
    return treatment

def _norm_meta(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

def _is_inscopix_path(relative_path: str) -> bool:
    return any(part.lower().startswith("inscopix_") for part in str(relative_path).split(os.sep))

def _find_acceleration_file(
    filenames: list[str],
    trace_meta: dict[str, Any],
) -> tuple[str | None, str]:
    accel_files = [name for name in filenames if ACCEL_RE.match(name)]
    if not accel_files:
        return None, "missing"

    trace_key = _norm_meta(trace_meta["filename_metadata"])
    exact = [
        name
        for name in accel_files
        if _norm_meta(ACCEL_RE.match(name).group("meta")) == trace_key  # type: ignore[union-attr]
    ]
    if exact:
        return sorted(exact)[0], "exact"

    mouse_alias = trace_meta.get("mouse_alias")
    if mouse_alias is not None:
        same_mouse = []
        for name in accel_files:
            parsed = parse_file_metadata(name)
            if parsed and parsed.get("mouse_alias") == mouse_alias:
                same_mouse.append(name)
        if len(same_mouse) == 1:
            return same_mouse[0], "mouse_alias"

    if len(accel_files) == 1:
        return accel_files[0], "single_acceleration_in_folder"

    return sorted(accel_files)[0], "ambiguous_first_sorted"

def discover_paper2_datasets(
    root: str,
    include_old: bool = False,
    allow_inscopix_without_acceleration: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Find paper-2 vessel datasets by scanning filenames under *root*."""
    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    if not os.path.isdir(root):
        return pd.DataFrame(), [f"Data folder does not exist: {root}"]

    for dirpath, dirnames, filenames in os.walk(root):
        if not include_old:
            dirnames[:] = [name for name in dirnames if name.lower() != "old"]

        for filename in filenames:
            match = TRACE_RE.match(filename)
            if not match:
                continue

            is_pivotable = "pivotable" in filename.lower()
            if is_pivotable and not include_old:
                continue

            trace_meta = parse_file_metadata(filename)
            if not trace_meta:
                warnings.append(f"Could not parse trace filename: {filename}")
                continue

            trace_path = os.path.join(dirpath, filename)
            rel_trace = os.path.relpath(trace_path, root)
            study, study_label = _study_label(rel_trace)
            folder_name = os.path.basename(dirpath)
            accel_name, accel_match = _find_acceleration_file(filenames, trace_meta)
            no_accel_allowed = (
                not accel_name
                and allow_inscopix_without_acceleration
                and _is_inscopix_path(rel_trace)
            )
            if no_accel_allowed:
                accel_match = "not_required_inscopix"
            accel_path = os.path.join(dirpath, accel_name) if accel_name else ""

            animal_id = parse_folder_animal_id(
                folder_name,
                trace_meta["mouse_alias"],
                trace_meta["treatment_raw"],
            )
            dataset_id = folder_name or trace_meta["filename_metadata"]

            records.append(
                {
                    "study": study,
                    "study_label": study_label,
                    "source_folder": folder_name,
                    "dataset_id": dataset_id,
                    "mouse_alias": trace_meta["mouse_alias"],
                    "animal_id": animal_id,
                    "treatment": paper2_contextual_treatment(
                        trace_meta["treatment_raw"],
                        study_label,
                    ),
                    "treatment_raw": trace_meta["treatment_raw"],
                    "filename_metadata": trace_meta["filename_metadata"],
                    "trace_file": filename,
                    "acceleration_file": accel_name or "",
                    "acceleration_match": accel_match,
                    "requires_acceleration": not no_accel_allowed,
                    "loadable": bool(accel_name) or no_accel_allowed,
                    "trace_path": trace_path,
                    "acceleration_path": accel_path,
                    "relative_trace_path": rel_trace,
                    "relative_acceleration_path": os.path.relpath(accel_path, root)
                    if accel_path
                    else "",
                    "has_acceleration": bool(accel_name),
                    "acceleration_missing_allowed": no_accel_allowed,
                    "is_pivotable": is_pivotable,
                    "file_size": os.path.getsize(trace_path),
                    "modified": pd.to_datetime(os.path.getmtime(trace_path), unit="s"),
                }
            )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        return manifest, warnings

    # A single missing numeric alias makes pandas infer float64 for the whole
    # column (for example 28 becomes 28.0). Keep aliases as stable strings so
    # display labels and filename-based vessel-map lookups remain correct.
    manifest["mouse_alias"] = manifest["mouse_alias"].map(canonical_mouse_alias)

    manifest["pair_key"] = (
        manifest["study_label"].astype(str)
        + "|"
        + manifest["source_folder"].astype(str)
        + "|"
        + manifest["mouse_alias"].astype(str)
        + "|"
        + manifest["treatment"].astype(str)
    )
    order = manifest.sort_values(
        [
            "pair_key",
            "loadable",
            "has_acceleration",
            "is_pivotable",
            "file_size",
            "modified",
        ],
        ascending=[True, False, False, True, False, False],
    )
    preferred_idx = order.groupby("pair_key", sort=False).head(1).index
    manifest["preferred"] = manifest.index.isin(preferred_idx)
    manifest = manifest.sort_values(
        ["study_label", "mouse_alias", "treatment", "relative_trace_path"]
    )
    return manifest.reset_index(drop=True), warnings

def load_paper2_vessel_data(
    root: str,
    manifest: pd.DataFrame | None = None,
    study_labels: list[str] | None = None,
    treatments: list[str] | None = None,
    preferred_only: bool = True,
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame, list[str]]:
    """Load selected vessel and acceleration CSVs into the Vessels structure."""
    if manifest is None:
        manifest, warnings = discover_paper2_datasets(root)
    else:
        manifest = manifest.copy()
        warnings = []

    if manifest.empty:
        return {}, manifest, warnings

    selected = manifest.copy()
    if preferred_only and "preferred" in selected.columns:
        selected = selected[selected["preferred"]].copy()
    if study_labels:
        selected = selected[selected["study_label"].isin(study_labels)].copy()
    if treatments:
        selected = selected[selected["treatment"].isin(treatments)].copy()

    Vessels: dict[str, dict[str, dict[str, Any]]] = {}
    loaded_rows: list[dict[str, Any]] = []
    seen_ids: dict[tuple[str, str], int] = {}

    for _, row in selected.iterrows():
        is_loadable = bool(row.get("loadable", row.get("has_acceleration", False)))
        if not is_loadable:
            warnings.append(f"Missing acceleration file for {row['relative_trace_path']}")
            continue
        group = str(row["treatment"])
        dataset_id = str(row["dataset_id"])
        key = (group, dataset_id)
        if key in seen_ids:
            seen_ids[key] += 1
            dataset_id = f"{dataset_id}#{seen_ids[key]}"
        else:
            seen_ids[key] = 1

        try:
            if not os.path.exists(row["trace_path"]):
                warnings.append(
                    f"Trace file no longer exists for {row['relative_trace_path']}. "
                    "Click Scan filenames again and reload the selected CSVs."
                )
                continue
            vessel_data = pd.read_csv(row["trace_path"], sep=",")
            if bool(row.get("has_acceleration", False)):
                if not os.path.exists(row["acceleration_path"]):
                    warnings.append(
                        f"Acceleration file no longer exists for {row['relative_acceleration_path']}. "
                        "Click Scan filenames again and reload the selected CSVs."
                    )
                    continue
                accel_data = pd.read_csv(row["acceleration_path"], sep=",", low_memory=False)
            else:
                accel_data = pd.DataFrame()
        except Exception as exc:
            warnings.append(f"Failed to load {row['relative_trace_path']}: {exc}")
            continue

        missing_columns = [
            column for column in REQUIRED_VESSEL_COLUMNS if column not in vessel_data.columns
        ]

        Vessels.setdefault(group, {})[dataset_id] = {
            "vessel_data": vessel_data,
            "acceleration_data": accel_data,
            "animal_id": row.get("animal_id", ""),
            "mouse_alias": row.get("mouse_alias", ""),
            "study": row.get("study", ""),
            "study_label": row.get("study_label", ""),
            "source_folder": row.get("source_folder", ""),
            "trace_file": row.get("trace_file", ""),
            "acceleration_file": row.get("acceleration_file", ""),
            "requires_acceleration": row.get("requires_acceleration", True),
            "acceleration_available": row.get("has_acceleration", False),
            "acceleration_missing_allowed": row.get("acceleration_missing_allowed", False),
            "trace_path": row.get("trace_path", ""),
            "acceleration_path": row.get("acceleration_path", ""),
            "filename_metadata": row.get("filename_metadata", ""),
            "missing_required_columns": missing_columns,
        }

        loaded = row.to_dict()
        loaded["loaded_dataset_id"] = dataset_id
        loaded["has_required_columns"] = len(missing_columns) == 0
        loaded["missing_required_columns"] = ", ".join(missing_columns)
        loaded["available_columns"] = ", ".join(str(column) for column in vessel_data.columns)
        loaded["vessel_rows"] = len(vessel_data)
        loaded["acceleration_rows"] = len(accel_data)
        loaded["requires_acceleration"] = bool(row.get("requires_acceleration", True))
        loaded["acceleration_missing_allowed"] = bool(row.get("acceleration_missing_allowed", False))
        loaded["requires_acceleration"] = bool(row.get("requires_acceleration", True))
        loaded["acceleration_missing_allowed"] = bool(row.get("acceleration_missing_allowed", False))
        loaded["unique_vessels"] = (
            vessel_data["Vessel ID"].nunique() if "Vessel ID" in vessel_data.columns else np.nan
        )
        loaded_rows.append(loaded)

    loaded_manifest = pd.DataFrame(loaded_rows)
    return Vessels, loaded_manifest, warnings

def _source_folder_mouse_alias(source_folder: Any, fallback: Any = "") -> str:
    match = re.match(r"^\s*(\d+)", str(source_folder))
    if match:
        return match.group(1)
    if pd.isna(fallback):
        return ""
    return str(fallback)

def _source_folder_treatment(source_folder: Any) -> str:
    tokens = [token for token in re.split(r"_+", str(source_folder)) if token]
    if not tokens:
        return ""
    if len(tokens) >= 4 and str(tokens[0]).isdigit():
        return canonical_treatment("_".join(tokens[3:]))
    return canonical_treatment(tokens[-1])

def _source_folder_treatment_for_study(source_folder: Any, study_label: Any) -> str:
    treatment = _source_folder_treatment(source_folder)
    treatment = paper2_contextual_treatment(treatment, str(study_label))
    if str(study_label) == PACAP_DOSING_STUDY_LABEL and treatment == "PACAP":
        return PACAP_DOSING_PLAIN_TREATMENT
    return treatment

def audit_paper2_dataset_naming(
    Vessels: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Flag folder/filename metadata mismatches that can affect pairing or grouping."""
    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for group, datasets in Vessels.items():
        for dataset_id, payload in datasets.items():
            source_folder = payload.get("source_folder", "")
            filename_mouse_alias = payload.get("mouse_alias", "")
            folder_mouse_alias = _source_folder_mouse_alias(source_folder, filename_mouse_alias)
            study_label = payload.get("study_label", "")
            folder_treatment = _source_folder_treatment_for_study(source_folder, study_label)
            group_treatment = paper2_contextual_treatment(group, study_label)
            base = {
                "Study": study_label,
                "Group from filename": group,
                "Dataset ID": dataset_id,
                "Pair mouse alias": folder_mouse_alias,
                "Filename mouse alias": filename_mouse_alias,
                "Animal ID": payload.get("animal_id", ""),
                "Source folder": source_folder,
                "Trace file": payload.get("trace_file", ""),
                "Trace path": payload.get("trace_path", ""),
            }
            flat_rows.append(base)

            if folder_mouse_alias and str(folder_mouse_alias) != str(filename_mouse_alias):
                rows.append(
                    {
                        **base,
                        "Issue": "Folder/file mouse alias mismatch",
                        "Details": (
                            f"Folder starts with mouse {folder_mouse_alias}, but trace filename "
                            f"was parsed as mouse {filename_mouse_alias}."
                        ),
                    }
                )

            if folder_treatment and folder_treatment != group_treatment:
                rows.append(
                    {
                        **base,
                        "Issue": "Folder/file treatment mismatch",
                        "Details": (
                            f"Folder suggests {folder_treatment}, but trace filename/group is "
                            f"{group_treatment}."
                        ),
                    }
                )

            if folder_mouse_alias and not str(payload.get("animal_id", "")).strip():
                rows.append(
                    {
                        **base,
                        "Issue": "Animal ID could not be parsed from folder",
                        "Details": (
                            "The folder and trace filename metadata do not line up cleanly enough "
                            "to extract the middle animal identifier."
                        ),
                    }
                )

    flat = pd.DataFrame(flat_rows)
    if not flat.empty:
        duplicate_keys = (
            flat.groupby(["Study", "Group from filename", "Pair mouse alias"], dropna=False)
            .size()
            .reset_index(name="n_datasets")
        )
        duplicate_keys = duplicate_keys[duplicate_keys["n_datasets"] > 1]
        for _, dup in duplicate_keys.iterrows():
            sub = flat[
                (flat["Study"] == dup["Study"])
                & (flat["Group from filename"] == dup["Group from filename"])
                & (flat["Pair mouse alias"] == dup["Pair mouse alias"])
            ]
            rows.append(
                {
                    "Study": dup["Study"],
                    "Group from filename": dup["Group from filename"],
                    "Dataset ID": ", ".join(sub["Dataset ID"].astype(str)),
                    "Pair mouse alias": dup["Pair mouse alias"],
                    "Filename mouse alias": ", ".join(sub["Filename mouse alias"].astype(str)),
                    "Animal ID": ", ".join(sub["Animal ID"].astype(str)),
                    "Source folder": ", ".join(sub["Source folder"].astype(str)),
                    "Trace file": ", ".join(sub["Trace file"].astype(str)),
                    "Trace path": "",
                    "Issue": "Duplicate dataset for same mouse/condition",
                    "Details": (
                        f"{int(dup['n_datasets'])} datasets share the same study, condition, "
                        "and pair mouse alias."
                    ),
                }
            )

    return pd.DataFrame(rows)

def audit_paper2_pairing_coverage(
    Vessels: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Show mouse aliases missing one or more treatments within each study."""
    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for group, datasets in Vessels.items():
        for dataset_id, payload in datasets.items():
            source_folder = payload.get("source_folder", "")
            flat_rows.append(
                {
                    "Study": payload.get("study_label", ""),
                    "Pair mouse alias": _source_folder_mouse_alias(
                        source_folder,
                        payload.get("mouse_alias", ""),
                    ),
                    "Group": canonical_treatment(group),
                    "Dataset ID": dataset_id,
                    "Source folder": source_folder,
                    "Trace file": payload.get("trace_file", ""),
                }
            )

    flat = pd.DataFrame(flat_rows)
    if flat.empty:
        return flat

    for study, study_df in flat.groupby("Study", dropna=False):
        group_counts = study_df.groupby("Group", dropna=False)["Pair mouse alias"].nunique()
        groups = sorted(group_counts[group_counts >= 2].index.astype(str).tolist())
        if len(groups) < 2:
            continue
        for alias, alias_df in study_df.groupby("Pair mouse alias", dropna=False):
            present = sorted(
                group
                for group in alias_df["Group"].dropna().astype(str).unique().tolist()
                if group in groups
            )
            missing = [group for group in groups if group not in present]
            if missing:
                rows.append(
                    {
                        "Study": study,
                        "Pair mouse alias": alias,
                        "Present groups": ", ".join(present),
                        "Missing groups if paired": ", ".join(missing),
                        "Expected groups in study": ", ".join(groups),
                        "Dataset IDs": ", ".join(alias_df["Dataset ID"].astype(str)),
                        "Source folders": ", ".join(alias_df["Source folder"].astype(str)),
                    }
                )

    return pd.DataFrame(rows)

def validate_paper2_vessel_data(
    Vessels: dict[str, dict[str, dict[str, Any]]],
    gap_threshold_sec: float = 1.5,
    accepted_only: bool = True,
    min_response_windows: int = 2,
) -> pd.DataFrame:
    """Return selected datasets that need fixing before segmentation."""
    issue_rows: list[dict[str, Any]] = []
    min_response_windows = max(1, int(min_response_windows))

    for group, datasets in Vessels.items():
        for dataset_id, payload in datasets.items():
            vessel_data = payload.get("vessel_data", pd.DataFrame())
            base = {
                "Group": group,
                "Dataset ID": dataset_id,
                "Study": payload.get("study_label", ""),
                "Mouse alias": payload.get("mouse_alias", ""),
                "Source folder": payload.get("source_folder", ""),
                "Trace file": payload.get("trace_file", ""),
                "Trace path": payload.get("trace_path", ""),
            }

            missing_columns = [
                column for column in REQUIRED_VESSEL_COLUMNS if column not in vessel_data.columns
            ]
            if missing_columns:
                issue_rows.append(
                    {
                        **base,
                        "Issue": "Missing required column(s)",
                        "Details": ", ".join(missing_columns),
                        "Available columns": ", ".join(str(column) for column in vessel_data.columns),
                    }
                )
                continue

            work = vessel_data
            if accepted_only and "Vessel Status" in work.columns:
                work = work[work["Vessel Status"].astype(str).str.lower() == "accepted"]

            if work.empty:
                issue_rows.append(
                    {
                        **base,
                        "Issue": "No vessel rows available after filtering",
                        "Details": "No accepted vessels found" if accepted_only else "No vessel rows found",
                        "Available columns": ", ".join(str(column) for column in vessel_data.columns),
                    }
                )
                continue

            first_vessel_id = next(iter(work.groupby("Vessel ID").groups), None)
            if first_vessel_id is None:
                issue_rows.append(
                    {
                        **base,
                        "Issue": "No vessel IDs found",
                        "Details": "The Vessel ID column is present but has no groups",
                        "Available columns": ", ".join(str(column) for column in vessel_data.columns),
                    }
                )
                continue

            first_vessel = work[work["Vessel ID"] == first_vessel_id].sort_values("Time (s)")
            time_values = first_vessel["Time (s)"].to_numpy(dtype=float)
            break_count = int(np.sum(np.diff(time_values) > gap_threshold_sec)) if len(time_values) > 1 else 0
            if break_count < min_response_windows:
                issue_rows.append(
                    {
                        **base,
                        "Issue": "Not enough phase breaks",
                        "Details": (
                            f"Detected {break_count} break(s) using gap threshold "
                            f"{gap_threshold_sec:g} sec; segmentation needs at least "
                            f"{min_response_windows}"
                        ),
                        "Available columns": ", ".join(str(column) for column in vessel_data.columns),
                    }
                )

    return pd.DataFrame(issue_rows)

def attach_dataset_metadata(
    Segmented_Vessels: dict[str, dict[str, dict[str, Any]]],
    Vessels: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Copy filename-derived dataset metadata onto segmented metrics."""
    metadata_keys = [
        "animal_id",
        "mouse_alias",
        "study",
        "study_label",
        "source_folder",
        "trace_file",
        "acceleration_file",
        "trace_path",
        "acceleration_path",
        "filename_metadata",
    ]
    for group, datasets in Segmented_Vessels.items():
        for dataset_id, metrics in datasets.items():
            payload = Vessels.get(group, {}).get(dataset_id, {})
            for key in metadata_keys:
                metrics[key] = payload.get(key, "")
    return Segmented_Vessels

def classify_vessel_sizes(baseline_diameters_um: Any) -> np.ndarray:
    baseline_diameters_um = np.asarray(baseline_diameters_um, dtype=float)
    out = np.full(baseline_diameters_um.shape, "Unknown", dtype=object)
    valid = ~np.isnan(baseline_diameters_um)
    out[valid & (baseline_diameters_um <= SIZE_SMALL_MAX_UM)] = "Small"
    out[valid & (baseline_diameters_um > SIZE_SMALL_MAX_UM) & (baseline_diameters_um < SIZE_LARGE_MIN_UM)] = (
        SIZE_MIDDLE_LABEL
    )
    out[valid & (baseline_diameters_um >= SIZE_LARGE_MIN_UM)] = "Large"
    return out

def _align_len(arr: Any, n: int) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    if values.size == n:
        return values
    if values.size > n:
        return values[:n]
    return np.concatenate([values, np.full(n - values.size, np.nan)])

def _finite_int(value: Any) -> int | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return int(round(numeric))

def _response_frame_limit(metrics: dict[str, Any], response_windows: list[Any]) -> int | None:
    direct = _finite_int(metrics.get("response_frames"))
    if direct is not None and direct > 0:
        return direct

    response_sec = _finite_int(metrics.get("response_sec"))
    sampling_rate = metrics.get("sampling_rate")
    try:
        sampling_rate_value = float(sampling_rate)
    except Exception:
        sampling_rate_value = np.nan
    if response_sec is not None and np.isfinite(sampling_rate_value) and sampling_rate_value > 0:
        return max(1, int(round(float(response_sec) * sampling_rate_value)))

    candidate_lengths: list[int] = []
    for frame_key in ["resp1_frames", "resp2_frames"]:
        try:
            frame_start, frame_end = metrics.get(frame_key, (np.nan, np.nan))
        except Exception:
            continue
        frame_start = _finite_int(frame_start)
        frame_end = _finite_int(frame_end)
        if frame_start is not None and frame_end is not None and frame_end > frame_start:
            candidate_lengths.append(frame_end - frame_start)

    for window in response_windows[:2]:
        if not isinstance(window, dict):
            continue
        frame_start = _finite_int(window.get("start_frame", np.nan))
        frame_end = _finite_int(window.get("end_frame", np.nan))
        if frame_start is not None and frame_end is not None and frame_end > frame_start:
            candidate_lengths.append(frame_end - frame_start)

    if candidate_lengths:
        return max(1, int(round(float(np.nanmedian(candidate_lengths)))))
    return None

def _window_values_from_frames(
    metrics: dict[str, Any],
    frame_start: Any,
    frame_end: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    trace_matrix = np.asarray(metrics.get("raw_trace_matrix", []), dtype=float)
    if trace_matrix.ndim != 2 or trace_matrix.shape[1] == 0:
        return None

    start = _finite_int(frame_start)
    end = _finite_int(frame_end)
    if start is None or end is None:
        return None

    n_frames = trace_matrix.shape[0]
    start = max(0, min(start, n_frames))
    end = max(start, min(end, n_frames))
    if end <= start:
        n_vessels = trace_matrix.shape[1]
        empty = np.full(n_vessels, np.nan)
        return empty, empty, empty

    try:
        baseline_start, baseline_end = metrics.get("baseline_frames", (0, min(n_frames, 900)))
    except Exception:
        baseline_start, baseline_end = (0, min(n_frames, 900))
    baseline_start = max(0, min(int(baseline_start), n_frames))
    baseline_end = max(baseline_start, min(int(baseline_end), n_frames))
    if baseline_end <= baseline_start:
        return None

    baseline = trace_matrix[baseline_start:baseline_end, :]
    baseline_mean = np.nanmean(baseline, axis=0)
    baseline_sd = np.nanstd(baseline, axis=0)
    baseline_sd = np.where(baseline_sd == 0, np.nan, baseline_sd)
    response = trace_matrix[start:end, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        z_trace = (trace_matrix - baseline_mean) / baseline_sd
        response_mean = np.nanmean(response, axis=0)
        abs_change = response_mean - baseline_mean
        percent_change = (abs_change / baseline_mean) * 100

    z_values = np.nanmean(z_trace[start:end, :], axis=0)
    return z_values, percent_change, abs_change

def _iter_response_window_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    response_windows = metrics.get("response_windows", [])
    if response_windows:
        specs: list[dict[str, Any]] = []
        frame_limit = _response_frame_limit(metrics, response_windows)
        mode = str(metrics.get("response_window_mode", ""))
        fixed_mode = mode != "all_gap_bounded"
        for idx, window in enumerate(response_windows, start=1):
            window_order = int(window.get("window_order", idx))
            frame_start = window.get("start_frame", np.nan)
            frame_end = window.get("end_frame", np.nan)
            start_int = _finite_int(frame_start)
            end_int = _finite_int(frame_end)
            if fixed_mode and frame_limit is not None and start_int is not None:
                fixed_end = start_int + int(frame_limit)
                if end_int is None:
                    frame_end = fixed_end
                else:
                    frame_end = min(end_int, fixed_end)

            values = _window_values_from_frames(metrics, frame_start, frame_end)
            if values is None:
                z_values = window.get("z_response", [])
                percent_values = window.get("percent_change", [])
                abs_values = window.get("abs_um_change", [])
            else:
                z_values, percent_values, abs_values = values

            specs.append(
                {
                    "window": window.get("name", f"Response {window_order}"),
                    "window_order": window_order,
                    "z_values": z_values,
                    "percent_values": percent_values,
                    "abs_values": abs_values,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                }
            )
        return specs

    fallback = [
        (
            "Response 1",
            1,
            "z_response1",
            "percent_change_resp1",
            "abs_um_change_resp1",
            "resp1_frames",
        ),
        (
            "Response 2",
            2,
            "z_response2",
            "percent_change_resp2",
            "abs_um_change_resp2",
            "resp2_frames",
        ),
    ]
    specs = []
    for window, window_order, z_key, pct_key, abs_key, frame_key in fallback:
        frame_start, frame_end = metrics.get(frame_key, (np.nan, np.nan))
        values = _window_values_from_frames(metrics, frame_start, frame_end)
        if values is None:
            z_values = metrics.get(z_key, [])
            percent_values = metrics.get(pct_key, [])
            abs_values = metrics.get(abs_key, [])
        else:
            z_values, percent_values, abs_values = values
        specs.append(
            {
                "window": window,
                "window_order": window_order,
                "z_values": z_values,
                "percent_values": percent_values,
                "abs_values": abs_values,
                "frame_start": frame_start,
                "frame_end": frame_end,
            }
        )
    return specs

def build_z_response_window_df(
    Segmented_Vessels: dict[str, dict[str, dict[str, Any]]],
    target_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Build one row per vessel per response window using baseline-z responses."""
    groups = set(target_groups or Segmented_Vessels.keys())
    rows: list[dict[str, Any]] = []

    for group, datasets in Segmented_Vessels.items():
        if group not in groups:
            continue

        for dataset_id, metrics in datasets.items():
            trace_matrix = np.asarray(metrics["raw_trace_matrix"], dtype=float)
            baseline_start, baseline_end = metrics["baseline_frames"]
            baseline_um = np.nanmean(trace_matrix[baseline_start:baseline_end, :], axis=0)
            n_vessels = baseline_um.size

            for spec in _iter_response_window_specs(metrics):
                z_values = _align_len(spec["z_values"], n_vessels)
                pct_values = _align_len(spec["percent_values"], n_vessels)
                abs_values = _align_len(spec["abs_values"], n_vessels)
                frame_start = spec["frame_start"]
                frame_end = spec["frame_end"]

                for vessel_index in range(n_vessels):
                    rows.append(
                        {
                            "group": group,
                            "dataset_id": dataset_id,
                            "mouse_alias": metrics.get("mouse_alias", ""),
                            "animal_id": metrics.get("animal_id", ""),
                            "study": metrics.get("study_label", metrics.get("study", "")),
                            "source_folder": metrics.get("source_folder", ""),
                            "trace_file": metrics.get("trace_file", ""),
                            "vessel_index": vessel_index,
                            "baseline_um": baseline_um[vessel_index],
                            "window": spec["window"],
                            "window_order": spec["window_order"],
                            "window_start_frame": frame_start,
                            "window_end_frame": frame_end,
                            "z_response": z_values[vessel_index],
                            "percent_change": pct_values[vessel_index],
                            "abs_um_change": abs_values[vessel_index],
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    numeric_cols = [
        "baseline_um",
        "window_start_frame",
        "window_end_frame",
        "z_response",
        "percent_change",
        "abs_um_change",
    ]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)

def classify_paper2_compartment_from_rgb(
    color_r: Any,
    color_g: Any,
    color_b: Any,
) -> str:
    """Classify Paper 2 vessel-line colors: green is pia, red is dura."""
    try:
        r = int(float(color_r))
        g = int(float(color_g))
        b = int(float(color_b))
    except Exception:
        return "unknown"

    if r >= 200 and g <= 100 and b <= 100:
        return "dura"
    if g >= 120 and g > r and g > b:
        return "pia"
    return "unknown"

def _rgb_text(values: tuple[Any, Any, Any] | None) -> str:
    if values is None:
        return ""
    try:
        return f"({int(float(values[0]))}, {int(float(values[1]))}, {int(float(values[2]))})"
    except Exception:
        return ""

def _first_number(value: Any) -> str | None:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else None

def _find_vessel_lines_path(payload: dict[str, Any]) -> str | None:
    trace_path = payload.get("trace_path", "")
    if trace_path:
        folder = os.path.dirname(str(trace_path))
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.lower().startswith("vessel lines_") and name.lower().endswith(".csv"):
                    return os.path.join(folder, name)
    return None

def _vessel_order_from_payload(payload: dict[str, Any]) -> list[str]:
    vessel_data = payload.get("vessel_data", pd.DataFrame())
    if isinstance(vessel_data, pd.DataFrame) and "Vessel ID" in vessel_data.columns:
        return [str(vessel_id).strip() for vessel_id, _ in vessel_data.groupby("Vessel ID")]
    return []

def _load_compartment_labels_for_payload(
    payload: dict[str, Any],
    vessel_order: list[str],
) -> tuple[list[str], list[str], str, list[str]]:
    warnings: list[str] = []
    lines_path = _find_vessel_lines_path(payload)
    if lines_path is None:
        return ["unknown"] * len(vessel_order), [""] * len(vessel_order), "", [
            "Missing Vessel lines CSV"
        ]

    try:
        lines_df = pd.read_csv(lines_path)
    except Exception as exc:
        return ["unknown"] * len(vessel_order), [""] * len(vessel_order), lines_path, [
            f"Could not read Vessel lines CSV: {exc}"
        ]

    lines_df.columns = [str(column).strip() for column in lines_df.columns]
    required = {"Name", "ColorR", "ColorG", "ColorB"}
    if not required.issubset(lines_df.columns):
        missing = ", ".join(sorted(required - set(lines_df.columns)))
        return ["unknown"] * len(vessel_order), [""] * len(vessel_order), lines_path, [
            f"Missing Vessel lines column(s): {missing}"
        ]

    if "Status" in lines_df.columns:
        lines_df = lines_df[
            lines_df["Status"].astype(str).str.lower().str.strip() == "accepted"
        ].copy()

    name_to_info: dict[str, tuple[str, str]] = {}
    number_to_info: dict[str, tuple[str, str]] = {}
    for _, row in lines_df.iterrows():
        name = str(row["Name"]).strip()
        rgb = (row["ColorR"], row["ColorG"], row["ColorB"])
        compartment = classify_paper2_compartment_from_rgb(*rgb)
        info = (compartment, _rgb_text(rgb))
        name_to_info[name] = info
        number = _first_number(name)
        if number is not None and number not in number_to_info:
            number_to_info[number] = info

    labels: list[str] = []
    rgbs: list[str] = []
    for vessel_id in vessel_order:
        info = name_to_info.get(str(vessel_id).strip())
        if info is None:
            number = _first_number(vessel_id)
            info = number_to_info.get(number) if number is not None else None
        if info is None:
            labels.append("unknown")
            rgbs.append("")
        else:
            labels.append(info[0])
            rgbs.append(info[1])

    if labels and all(label == "unknown" for label in labels):
        warnings.append("No accepted vessels were red or green in Vessel lines CSV")

    return labels, rgbs, lines_path, warnings

def build_vessel_qc_df(
    Segmented_Vessels: dict[str, dict[str, dict[str, Any]]],
    Vessels: dict[str, dict[str, dict[str, Any]]],
    target_groups: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per accepted segmented vessel for Paper 2 QC counts."""
    groups = set(target_groups or Segmented_Vessels.keys())
    rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    for group, datasets in Segmented_Vessels.items():
        if group not in groups:
            continue

        for dataset_id, metrics in datasets.items():
            payload = Vessels.get(group, {}).get(dataset_id, {})
            trace_matrix = np.asarray(metrics.get("raw_trace_matrix", []), dtype=float)
            if trace_matrix.ndim != 2 or trace_matrix.shape[1] == 0:
                continue

            n_frames, n_vessels = trace_matrix.shape
            baseline_start, baseline_end = metrics.get(
                "baseline_frames",
                (0, min(n_frames, 1)),
            )
            baseline_start = max(0, min(int(baseline_start), n_frames))
            baseline_end = max(baseline_start, min(int(baseline_end), n_frames))
            baseline_um = np.nanmean(trace_matrix[baseline_start:baseline_end, :], axis=0)
            vessel_sizes = classify_vessel_sizes(baseline_um)

            vessel_order = _vessel_order_from_payload(payload)
            if len(vessel_order) < n_vessels:
                vessel_order = vessel_order + [
                    f"index_{idx}" for idx in range(len(vessel_order), n_vessels)
                ]
            else:
                vessel_order = vessel_order[:n_vessels]

            labels, rgbs, lines_path, label_warnings = _load_compartment_labels_for_payload(
                payload,
                vessel_order,
            )
            if len(labels) < n_vessels:
                labels = labels + ["unknown"] * (n_vessels - len(labels))
            else:
                labels = labels[:n_vessels]
            if len(rgbs) < n_vessels:
                rgbs = rgbs + [""] * (n_vessels - len(rgbs))
            else:
                rgbs = rgbs[:n_vessels]

            metadata = {
                "group": group,
                "dataset_id": dataset_id,
                "mouse_alias": metrics.get("mouse_alias", payload.get("mouse_alias", "")),
                "animal_id": metrics.get("animal_id", payload.get("animal_id", "")),
                "study": metrics.get(
                    "study_label",
                    metrics.get("study", payload.get("study_label", payload.get("study", ""))),
                ),
                "source_folder": metrics.get(
                    "source_folder",
                    payload.get("source_folder", ""),
                ),
                "trace_file": metrics.get("trace_file", payload.get("trace_file", "")),
                "vessel_lines_file": lines_path,
                "baseline_start_frame": baseline_start,
                "baseline_end_frame": baseline_end,
            }

            for warning in label_warnings:
                warning_rows.append({**metadata, "warning": warning})

            for vessel_index in range(n_vessels):
                rows.append(
                    {
                        **metadata,
                        "vessel_index": vessel_index,
                        "vessel_id": vessel_order[vessel_index],
                        "baseline_um": baseline_um[vessel_index],
                        "vessel_size": str(vessel_sizes[vessel_index]),
                        "compartment": str(labels[vessel_index]),
                        "rgb": rgbs[vessel_index],
                    }
                )

    qc_df = pd.DataFrame(rows)
    if not qc_df.empty:
        qc_df["baseline_um"] = qc_df["baseline_um"].replace([np.inf, -np.inf], np.nan)

    return qc_df.reset_index(drop=True), pd.DataFrame(warning_rows)
