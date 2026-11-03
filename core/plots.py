# Vessel imaging analysis methods.
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.config import CATEGORY_COLORS, TREATMENT_COLORS

def _baseline_z(trace_matrix: np.ndarray, baseline_frames: tuple[int, int]) -> np.ndarray:
    start, end = baseline_frames
    baseline = trace_matrix[int(start) : int(end), :]
    mean = np.nanmean(baseline, axis=0)
    sd = np.nanstd(baseline, axis=0, ddof=0)
    sd[~np.isfinite(sd) | (sd == 0)] = np.nan
    return (trace_matrix - mean) / sd

def build_animal_timecourses(
    segmented: dict,
    qc_df: pd.DataFrame,
    category_mode: str = "all",
) -> pd.DataFrame:
    """Return one row per animal/session, category, segment, and original frame."""
    rows: list[dict] = []
    qc_lookup = {}
    if not qc_df.empty:
        for key, frame in qc_df.groupby(["group", "dataset_id"], sort=False):
            qc_lookup[(str(key[0]), str(key[1]))] = frame.sort_values("vessel_index")

    for group, datasets in segmented.items():
        for dataset_id, metrics in datasets.items():
            trace = np.asarray(metrics.get("raw_trace_matrix", []), dtype=float)
            if trace.ndim != 2 or trace.shape[1] == 0:
                continue
            z_trace = _baseline_z(trace, metrics.get("baseline_frames", (0, min(900, len(trace)))))
            n_vessels = z_trace.shape[1]
            labels = np.array(["All vessels"] * n_vessels, dtype=object)
            q = qc_lookup.get((str(group), str(dataset_id)))
            if q is not None and category_mode in {"size", "compartment"}:
                column = "vessel_size" if category_mode == "size" else "compartment"
                mapped = dict(zip(q["vessel_index"].astype(int), q[column].astype(str)))
                labels = np.array([mapped.get(i, "Unknown" if category_mode == "size" else "unknown") for i in range(n_vessels)])

            baseline_start, baseline_end = metrics.get("baseline_frames", (0, min(900, len(trace))))
            segments = [("Baseline", 0, int(baseline_end))]
            for window in metrics.get("response_windows", []):
                segments.append(
                    (
                        str(window.get("name", f"Response {window.get('window_order', '')}")),
                        int(window.get("start_frame", 0)),
                        int(window.get("end_frame", 0)),
                    )
                )

            for category in pd.unique(labels):
                indices = np.flatnonzero(labels == category)
                if indices.size == 0:
                    continue
                for segment_order, (segment, start, end) in enumerate(segments):
                    start = max(0, min(start, len(z_trace)))
                    end = max(start, min(end, len(z_trace)))
                    values = np.nanmean(z_trace[start:end, :][:, indices], axis=1)
                    for frame_in_segment, value in enumerate(values):
                        rows.append(
                            {
                                "study": metrics.get("study_label", metrics.get("study", "")),
                                "treatment": str(group),
                                "dataset_id": str(dataset_id),
                                "mouse_alias": str(metrics.get("mouse_alias", "")),
                                "category": str(category),
                                "segment": segment,
                                "segment_order": segment_order,
                                "frame_in_segment": frame_in_segment,
                                "z": value,
                                "n_vessels": int(indices.size),
                            }
                        )
    return pd.DataFrame(rows)

def bin_timecourses(animal_df: pd.DataFrame, bin_seconds: int, sampling_rate: float = 1.0) -> pd.DataFrame:
    if animal_df.empty:
        return animal_df.copy()
    out = animal_df.copy()
    frames_per_bin = max(1, int(round(float(bin_seconds) * float(sampling_rate))))
    out["visual_bin"] = (out["frame_in_segment"] // frames_per_bin).astype(int)
    keys = [
        "study",
        "treatment",
        "dataset_id",
        "mouse_alias",
        "category",
        "segment",
        "segment_order",
        "visual_bin",
    ]
    return (
        out.groupby(keys, observed=True, as_index=False)
        .agg(z=("z", "mean"), n_vessels=("n_vessels", "first"))
        .assign(frame_in_segment=lambda d: d["visual_bin"] * frames_per_bin + frames_per_bin / 2)
    )

def summarize_timecourses(animal_df: pd.DataFrame, sampling_rate: float, visual_gap_min: float) -> pd.DataFrame:
    if animal_df.empty:
        return animal_df.copy()
    work = animal_df.copy()
    segment_lengths = (
        work.groupby(["segment_order", "segment"], observed=True)["frame_in_segment"].max().reset_index()
    )
    offsets = {}
    cursor = 0.0
    for row in segment_lengths.sort_values("segment_order").itertuples(index=False):
        order = int(row.segment_order)
        if order == 0:
            length_min = (float(row.frame_in_segment) + 1) / sampling_rate / 60
            offsets[order] = -length_min
            cursor = visual_gap_min
        else:
            offsets[order] = cursor
            cursor += (float(row.frame_in_segment) + 1) / sampling_rate / 60 + visual_gap_min
    work["time_min"] = work.apply(
        lambda row: offsets[int(row["segment_order"])]
        + float(row["frame_in_segment"]) / sampling_rate / 60,
        axis=1,
    )
    keys = ["study", "treatment", "category", "segment", "segment_order", "time_min"]
    summary = (
        work.groupby(keys, observed=True, as_index=False)
        .agg(
            mean_z=("z", "mean"),
            sd=("z", "std"),
            animals=("mouse_alias", "nunique"),
            sessions=("dataset_id", "nunique"),
        )
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["sessions"].clip(lower=1))
    return summary

def timecourse_figure(
    summary: pd.DataFrame,
    color_by: str,
    y_range: tuple[float, float] | None,
    y_tick: float | None,
    title: str,
    challenge_timeline: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig
    series_col = "treatment" if color_by == "treatment" else "category"
    palette = TREATMENT_COLORS if color_by == "treatment" else CATEGORY_COLORS
    first_segment = int(summary["segment_order"].min())
    for series, series_df in summary.groupby(series_col, sort=False):
        color = palette.get(str(series), "#374151")
        for segment_order, segment_df in series_df.groupby("segment_order", sort=True):
            segment_df = segment_df.sort_values("time_min")
            x = segment_df["time_min"].to_numpy(float)
            mean = segment_df["mean_z"].to_numpy(float)
            sem = segment_df["sem"].fillna(0).to_numpy(float)
            show = int(segment_order) == first_segment
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.18),
                    line={"width": 0},
                    hoverinfo="skip",
                    legendgroup=str(series),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    name=str(series),
                    line={"color": color, "width": 2.5},
                    legendgroup=str(series),
                    showlegend=show,
                    customdata=np.column_stack(
                        [segment_df["sem"], segment_df["animals"], segment_df["sessions"], segment_df["segment"]]
                    ),
                    hovertemplate=(
                        ("" if challenge_timeline else "%{customdata[3]}<br>")
                        + "Time=%{x:.2f} min<br>Mean z=%{y:.3f}"
                        + "<br>SEM=%{customdata[0]:.3f}<br>Animals=%{customdata[1]}"
                        + "<br>Sessions=%{customdata[2]}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    for value, label in [(1.96, "+1.96"), (-1.96, "-1.96")]:
        fig.add_hline(y=value, line_dash="dot", line_color="#6B7280", annotation_text=label)
    fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF")
    if challenge_timeline:
        fig.add_vline(x=0, line_dash="dot", line_color="#6B7280", line_width=1.5)
        fig.add_annotation(
            x=0,
            y=1.02,
            yref="paper",
            text="Pharmacological challenge",
            showarrow=False,
            xanchor="left",
            font={"color": "#4B5563"},
        )
        baseline = summary[summary["segment_order"] == summary["segment_order"].min()]
        if not baseline.empty:
            baseline_midpoint = (
                float(baseline["time_min"].min()) + float(baseline["time_min"].max())
            ) / 2
            fig.add_annotation(
                x=baseline_midpoint, y=1.02, yref="paper", text="Baseline recording",
                showarrow=False, font={"color": "#6B7280"},
            )
    else:
        for _, frame in summary.groupby(["segment_order", "segment"], observed=True):
            x0 = float(frame["time_min"].min())
            fig.add_vline(x=x0, line_dash="dot", line_color="#94A3B8")
            fig.add_annotation(
                x=x0, y=1.02, yref="paper", text=str(frame["segment"].iloc[0]), showarrow=False
            )
    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Vasodilation (Z-score)",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="",
        height=520,
        margin={"l": 65, "r": 30, "t": 75, "b": 60},
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    if y_tick:
        fig.update_yaxes(dtick=float(y_tick))
    return fig

def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
