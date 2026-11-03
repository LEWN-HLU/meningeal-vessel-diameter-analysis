# Vessel imaging analysis methods.
from __future__ import annotations
import numpy as np
from typing import Any

def filter_accepted_vessels(
    Vessels: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Remove rejected vessels from all mice in-place, keeping only 'accepted' ones."""
    for group, mice in Vessels.items():
        for mouse_id, data in mice.items():
            df = data["vessel_data"]
            if "Vessel Status" not in df.columns:
                print(f"Warning: 'Vessel Status' not found in mouse {mouse_id} of group {group}")
                continue
            df_accepted = df[df["Vessel Status"].str.lower() == "accepted"].copy()
            Vessels[group][mouse_id]["vessel_data"] = df_accepted
    print("Rejected vessels removed successfully from all mice.")
    return Vessels

def segment_vessels(
    Vessels: dict[str, dict[int, dict[str, Any]]],
    baseline_sec: int = 15 * 60,
    response_sec: int = 10 * 60,
    sampling_rate: int = 1,
    gap_threshold_sec: float = 1.5,
    response_window_mode: str = "first_two_fixed",
    max_response_windows: int | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Segment vessel traces into baseline and response windows.

    Returns
    -------
    Segmented_Vessels : dict
        Nested dict with z-scores, absolute/percent changes, and raw trace matrices.
    """
    valid_modes = {"first_two_fixed", "all_fixed", "all_gap_bounded"}
    if response_window_mode not in valid_modes:
        raise ValueError(
            "response_window_mode must be one of: first_two_fixed, all_fixed, all_gap_bounded"
        )

    baseline_frames = int(baseline_sec * sampling_rate)
    response_frames = int(response_sec * sampling_rate)

    Segmented_Vessels = {}

    for group, mice in Vessels.items():
        Segmented_Vessels[group] = {}

        for mouse_id, data in mice.items():
            df = data["vessel_data"]
            grouped = df.groupby("Vessel ID")

            traces = []
            time_reference = None

            for _, vessel_df in grouped:
                vessel_df = vessel_df.sort_values("Time (s)")
                diameter = vessel_df["Diameter (um)"].values
                time = vessel_df["Time (s)"].values

                if time_reference is None:
                    time_reference = time

                traces.append(diameter)

            trace_matrix = np.column_stack(traces)
            total_frames = trace_matrix.shape[0]

            time_diffs = np.diff(time_reference)
            break_indices = np.where(time_diffs > gap_threshold_sec)[0] + 1

            if len(break_indices) < 1:
                print(f"Warning: Not enough breaks for mouse {mouse_id} in group {group}. Skipping.")
                continue

            idx_baseline_start = 0
            idx_baseline_end = min(idx_baseline_start + baseline_frames, total_frames)

            baseline = trace_matrix[idx_baseline_start:idx_baseline_end, :]

            baseline_mean = np.nanmean(baseline, axis=0)
            baseline_std = np.nanstd(baseline, axis=0)

            z_trace = (trace_matrix - baseline_mean) / baseline_std
            if response_window_mode == "first_two_fixed":
                raw_windows = [
                    (int(start), min(int(start) + response_frames, total_frames))
                    for start in break_indices[:2]
                ]
            elif response_window_mode == "all_fixed":
                raw_windows = [
                    (int(start), min(int(start) + response_frames, total_frames))
                    for start in break_indices
                ]
            else:
                raw_windows = []
                for break_idx, start in enumerate(break_indices):
                    end = (
                        int(break_indices[break_idx + 1])
                        if break_idx + 1 < len(break_indices)
                        else total_frames
                    )
                    raw_windows.append((int(start), end))

            if max_response_windows is not None:
                raw_windows = raw_windows[: max(0, int(max_response_windows))]

            response_windows = []
            for window_order, (frame_start, frame_end) in enumerate(raw_windows, start=1):
                frame_start = max(0, min(frame_start, total_frames))
                frame_end = max(frame_start, min(frame_end, total_frames))
                if frame_end <= frame_start:
                    continue

                response = trace_matrix[frame_start:frame_end, :]
                z_response = np.nanmean(z_trace[frame_start:frame_end, :], axis=0)
                response_mean = np.nanmean(response, axis=0)
                abs_change = response_mean - baseline_mean
                percent_change = (abs_change / baseline_mean) * 100
                response_windows.append(
                    {
                        "name": f"Response {window_order}",
                        "window_order": window_order,
                        "start_frame": frame_start,
                        "end_frame": frame_end,
                        "n_frames": frame_end - frame_start,
                        "z_response": z_response,
                        "abs_um_change": abs_change,
                        "percent_change": percent_change,
                    }
                )

            empty_response = np.full(trace_matrix.shape[1], np.nan)
            legacy_windows = response_windows[:2]
            while len(legacy_windows) < 2:
                legacy_windows.append(
                    {
                        "start_frame": np.nan,
                        "end_frame": np.nan,
                        "z_response": empty_response,
                        "abs_um_change": empty_response,
                        "percent_change": empty_response,
                    }
                )

            Segmented_Vessels[group][mouse_id] = {
                "z_response1": legacy_windows[0]["z_response"],
                "z_response2": legacy_windows[1]["z_response"],
                "abs_um_change_resp1": legacy_windows[0]["abs_um_change"],
                "abs_um_change_resp2": legacy_windows[1]["abs_um_change"],
                "percent_change_resp1": legacy_windows[0]["percent_change"],
                "percent_change_resp2": legacy_windows[1]["percent_change"],
                "mean_ztrace": np.nanmean(z_trace, axis=1),
                "raw_trace_matrix": trace_matrix,
                "baseline_frames": (idx_baseline_start, idx_baseline_end),
                "resp1_frames": (
                    legacy_windows[0]["start_frame"],
                    legacy_windows[0]["end_frame"],
                ),
                "resp2_frames": (
                    legacy_windows[1]["start_frame"],
                    legacy_windows[1]["end_frame"],
                ),
                "response_windows": response_windows,
                "break_indices": [int(idx) for idx in break_indices],
                "response_window_mode": response_window_mode,
                "max_response_windows": max_response_windows,
                "response_sec": int(response_sec),
                "response_frames": int(response_frames),
                "sampling_rate": sampling_rate,
                "gap_threshold_sec": float(gap_threshold_sec),
            }

    print("All mice processed using baseline and response windows.")
    return Segmented_Vessels
