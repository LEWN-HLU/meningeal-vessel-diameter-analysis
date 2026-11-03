# Input formats

## Vessel traces

Long-form CSV columns:

| Column | Meaning |
| --- | --- |
| `Vessel ID` | Vessel-segment identifier within session |
| `Time (s)` | Recording time in seconds, retaining inter-recording gaps |
| `Diameter (um)` | Diameter in micrometres |
| `Vessel Status` | Annotation status; only `accepted` rows enter analysis |

The processing workflow expects 1-Hz exported traces. Timestamps and row counts must agree across vessels within a session. Mouse identifiers must be consistent across repeated conditions; separate mice must not share an identifier.

For automatic discovery, use study folders `Levcromakalim`, `NTG`, and `PACAP`; the latter contains `Dosing study` and `Anti-PACAP` subfolders. Store each animal/session in a separate subfolder. Sequential-dose sessions use an `Inscopix_` study folder.

Trace filenames begin `Vessel traces_`, followed by numeric mouse alias, underscore, treatment token and `.csv`. The legacy `Vesssel traces_` spelling is also recognized. Treatment tokens include `Levcro`, `Nitroglycerin`, `Veh`/`Vehicle`, `0_1mgkg_PACAP`, `0_3mgkg_PACAP`, `PACAP`, and `Anti-PACAP mAb`. A JSON mapping can translate other input labels to canonical names without changing input files. NTG tokens `GR1`/`GR2` map to vehicle/nitroglycerin in that study; a plain PACAP token in the acute dosing-study folder denotes 0.1 mg/kg.

The discovery workflow matches non-Inscopix trace files to `Acceleration_` CSV filenames with corresponding metadata. This is an input-file matching rule; acceleration values are not used in vascular response calculations in this package. Inscopix traces can be loaded without an acceleration file.

## Compartment annotations

Place one `Vessel lines_*.csv` beside each session's trace. Required columns are `Name`, `ColorR`, `ColorG`, and `ColorB`. If `Status` is present, only accepted annotation rows are used. `Name` should match `Vessel ID`; numeric-token matching is the fallback.

RGB metadata encodes the supplied morphological labels:

- Dura: R ≥ 200, G ≤ 100, B ≤ 100.
- Pia: G ≥ 120, G > R and G > B.
- Otherwise, or if no label matches: unknown.

The program reads these labels; it does not determine anatomical identity from vessel dynamics. If several vessel-line files are present, the loader selects the first matching file, so use one intended annotation file per session.

## Statistical function inputs

Treatment models use a table with `mouse_alias`, `dataset_id`, `group`, and `z_response`, plus `vessel_size` or `compartment` when applicable. `session_values` constructs these values from vessel-level response tables. `window_order=1` identifies Response 1.

Diameter models additionally require `baseline_um`. The vessel-level baseline-compartment model requires `paper_source` for the study/source fixed effect. `fit_pacap_compartment` accepts vessel-window response rows and performs its own mouse/compartment aggregation. See function definitions and `ANALYSES.md` for formulas and return values.

## Sequential-dose times

Optional timing CSV columns are `dataset_id`, `dose_label`, `recorded_time_min`, `admin_time_min`, and `baseline_duration_min`. Supply three distinct, increasing dose times per included session, using the recorded-video clock of the diameter trace.

The baseline ends at the first recorded dose time and extends backwards by the supplied duration, clipped at zero. A full-dose window ends at the next recorded dose time, or the end of recording for the final dose. These are descriptive sequential-dose summaries, not independent acute-dose effects. The baseline z-score mean is near zero by construction.
