# %% [markdown]
"""Publication pipeline for VS Code (Python/Jupyter extension) or the command line.

Set DATA_FOLDER and OUTPUT_FOLDER below to run the cells interactively.
CLI: python run_analysis.py --data-root /path/to/data --output /path/to/results
Optional sequential-dosing timing: --dose-timing /path/to/timing.csv
No data, private paths, or previous study outputs are embedded in this script.
"""
# %%
from pathlib import Path
import argparse
import json
from publication import Settings, COHORTS, prepare, manuscript_statistics, dosing_comparison, inscopix_doses, timecourses, archive
from core.plots import timecourse_figure
import pandas as pd

DATA_FOLDER = None  # Set to a local input folder, never commit private paths.
OUTPUT_FOLDER = None  # Set to a NEW results folder outside the code repository.

def run(data_root, output, settings=None, dose_timing=None, treatment_aliases=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output folder to avoid overwriting prior results.')
    output.mkdir(parents=True,exist_ok=True)
    aliases=json.loads(Path(treatment_aliases).read_text()) if treatment_aliases else None
    a=prepare(str(data_root),settings or Settings(),treatment_aliases=aliases)
    tables,notes=manuscript_statistics(a)
    tables.update(dosing_comparison(a['response_df']))
    if dose_timing:
        tables.update(inscopix_doses(a,pd.read_csv(dose_timing,dtype={'dataset_id':str})))
    for cohort in COHORTS:
        for mode in ['all','size','compartment']:
            animal,summary=timecourses(a,cohort,mode)
            if summary.empty: continue
            slug=cohort.replace(' ','_')+'_'+mode
            tables[slug+'_timecourse_session']=animal
            tables[slug+'_timecourse_summary']=summary
            for treatment in summary.treatment.unique():
                sub=summary[summary.treatment.eq(treatment)]
                fig=timecourse_figure(sub,'category',None,None,cohort+' — '+str(treatment),True)
                fig.write_html(str(output/(slug+'_'+str(treatment)+'.html')),include_plotlyjs='directory')
    (output/'results.zip').write_bytes(archive(tables,notes,a['settings']))
    print('Results saved to',output)
    return tables,notes

# %%
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Process vessel traces and generate treatment summaries and time courses.')
    parser.add_argument('--data-root')
    parser.add_argument('--output')
    parser.add_argument('--baseline-seconds',type=int,default=890)
    parser.add_argument('--response-seconds',type=int,default=590)
    parser.add_argument('--extreme-z',type=float,default=20.)
    parser.add_argument('--dose-timing')
    parser.add_argument('--treatment-aliases',help='Local JSON mapping private treatment labels to public canonical labels')
    args=parser.parse_args()
    data=args.data_root or DATA_FOLDER
    output=args.output or OUTPUT_FOLDER
    if data and output:
        run(data,output,Settings(baseline_sec=args.baseline_seconds,response_sec=args.response_seconds,extreme_z_threshold=args.extreme_z),args.dose_timing,args.treatment_aliases)
    else:
        parser.print_help()
