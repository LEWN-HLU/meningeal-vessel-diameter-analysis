"""Manuscript-only orchestration; shared by the UI and command-line script."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from io import BytesIO
import json
import platform
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from importlib.metadata import version

from core.pipeline import scan_data, load_selected, run_pipeline, pool_pacap_dosing_vehicle, prefer_dosing_pacap_for_overlapping_mice
from core.paper_helpers import session_values, response1, with_provenance, canonical_mouse, fit_pacap_compartment
from core.mixed import mixed_model_size_interaction, mixed_model_two_group_mouse_intercept, mixed_model_two_group_category_interaction
from core.stats import holm_adjust
from core.plots import build_animal_timecourses, bin_timecourses, summarize_timecourses

DOSING = 'PACAP / Dosing study'
ANTI = 'PACAP / Anti-PACAP'
COHORTS = ['Levcromakalim', 'Nitroglycerin', 'PACAP 0.1 dosing', 'Anti-PACAP study', 'Fig3h pooled PACAP']

@dataclass(frozen=True)
class Settings:
    baseline_sec: int = 890
    response_sec: int = 590
    sampling_rate: int = 1
    gap_threshold_sec: float = 1.5
    extreme_z_threshold: float | None = 20.0

def prepare(root: str, settings: Settings, manifest: pd.DataFrame | None = None, treatment_aliases: dict | None = None) -> dict:
    """Read locally supplied data; never search a corporate location by default."""
    if manifest is None:
        manifest, scan_warnings = scan_data(root, allow_inscopix_without_acceleration=True)
    else:
        scan_warnings = []
    if manifest.empty:
        raise ValueError('No eligible trace files were found. See INPUT_FORMAT.md.')
    manifest=manifest.copy()
    if treatment_aliases:
        for col in ['treatment','treatment_raw']:
            manifest[col]=manifest[col].replace(treatment_aliases)
    normal = manifest[manifest['study_label'].isin(['Levcromakalim','NTG',DOSING,ANTI])].copy()
    allowed={'Levcro','Nitroglycerin','Veh','Vehicle','0_1mgkg_PACAP','0_3mgkg_PACAP','PACAP','Anti-PACAP mAb'}
    unknown=set(normal.treatment.astype(str))-allowed
    if unknown:
        raise ValueError('Unrecognized treatment label(s). Supply a local treatment-alias mapping before analysis: '+', '.join(sorted(unknown)))
    vessels, loaded, load_warnings = load_selected(root, normal, [], [])
    for group, datasets in vessels.items():
        for dataset, payload in datasets.items():
            if 'Vessel Status' not in payload['vessel_data']:
                raise ValueError(f'Missing Vessel Status: {group}/{dataset}; accepted-only analysis cannot be established.')
    result = run_pipeline(vessels, **asdict(settings), max_response_windows=None,
                          skip_invalid_datasets=False, pool_pacap_vehicle=False, pool_pacap_dose=False)
    if not result['ready']:
        raise ValueError('Dataset validation failed. Resolve the reported input issues before analysis:\n'+result['issues'].to_string(index=False))
    result.update(manifest=manifest, loaded=loaded, load_warnings=load_warnings,
                  scan_warnings=scan_warnings, settings=asdict(settings))
    # This view is local to Fig. 3h and its baseline tests.
    pool_groups={'PACAP','0_1mgkg_PACAP'}
    pv, ps, pr, pq, pm = pool_pacap_dosing_vehicle(
        {g:v for g,v in result['vessels'].items() if g in pool_groups},
        {g:v for g,v in result['segmented'].items() if g in pool_groups},
        result['response_df'][result['response_df'].group.isin(pool_groups)],
        result['qc_df'][result['qc_df'].group.isin(pool_groups)],
        source_groups={'0_1mgkg_pacap'}, target_group='PACAP', suffix='__pooled_0_1mgkg',
        remove_source_sessions=True)
    pv, ps, pr, pq, audit, _ = prefer_dosing_pacap_for_overlapping_mice(pv, ps, pr, pq)
    result['pooled'] = dict(vessels=pv, segmented=ps, response_df=pr, qc_df=pq)
    result['pooling_manifest'], result['replacement_audit'] = pm, audit
    return result

def inscopix_doses(analysis: dict, timing: pd.DataFrame) -> dict[str,pd.DataFrame]:
    """S4a: explicit per-session timing, full-period Z scores; no inferential test."""
    from core.dose_helpers import analyze_manifest_session_doses
    required={'dataset_id','dose_label','recorded_time_min','admin_time_min','baseline_duration_min'}
    if not required.issubset(timing):
        raise ValueError('Dose timing columns required: '+', '.join(sorted(required)))
    rows=[]
    manifest=analysis['manifest']
    for dataset, group in timing.groupby('dataset_id',sort=False):
        matches=manifest[manifest.dataset_id.astype(str).eq(str(dataset)) & manifest.study_label.astype(str).str.contains('Inscopix',case=False)]
        if len(matches)!=1:
            raise ValueError(f'Expected one Inscopix trace for {dataset}, found {len(matches)}.')
        group=group.sort_values('recorded_time_min')
        if len(group)!=3 or group.recorded_time_min.duplicated().any():
            raise ValueError('Specify three distinct dose times per Inscopix session.')
        if group.baseline_duration_min.nunique()!=1 or float(group.baseline_duration_min.iloc[0])<=0:
            raise ValueError('Provide one positive baseline duration per session.')
        table=group.rename(columns={'dose_label':'Dose label','recorded_time_min':'Recorded video time min','admin_time_min':'Admin time min'})
        outputs=analyze_manifest_session_doses(matches.iloc[0],table,float(group.baseline_duration_min.iloc[0]),10.)
        summary=outputs[4]
        if not summary.empty:
            summary=summary[summary.category.eq('All vessels')].copy()
            summary['dataset_id']=str(dataset)
            rows.append(summary)
    if not rows:
        return {}
    sessions=pd.concat(rows,ignore_index=True)
    mice=sessions.groupby(['mouse_alias','dose_order','dose_label'],observed=True,as_index=False).animal_session_mean_z.mean()
    wide=mice.pivot(index='mouse_alias',columns='dose_label',values='animal_session_mean_z').reset_index()
    return {'S4a_session_values':sessions,'S4a_mouse_values':mice,'S4a_Prism_matched':wide,
            'S4a_descriptive':descriptive(mice,['dose_order','dose_label'],'animal_session_mean_z')}

def select(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    w = with_provenance(frame)
    study, treatment = w['original_study'].astype(str), w['original_treatment'].astype(str)
    if cohort == 'Levcromakalim':
        mask = study.str.contains('Levcro', case=False, na=False) & treatment.isin(['Levcro','Veh','Vehicle'])
    elif cohort == 'Nitroglycerin':
        mask = study.str.contains('NTG|Nitro', case=False, regex=True, na=False) & treatment.isin(['Nitroglycerin','Veh','Vehicle'])
    elif cohort == 'PACAP 0.1 dosing':
        mask = study.eq(DOSING) & treatment.isin(['0_1mgkg_PACAP','Veh','Vehicle'])
    elif cohort == 'Anti-PACAP study':
        mask = study.eq(ANTI) & treatment.isin(['PACAP','Anti-PACAP mAb'])
    elif cohort == 'Fig3h pooled PACAP':
        mask = w['study'].eq(ANTI) & w['group'].eq('PACAP')
    else:
        raise ValueError(cohort)
    return w.loc[mask].copy()

def selected_view(analysis: dict, cohort: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    a = analysis['pooled'] if cohort == 'Fig3h pooled PACAP' else analysis
    q = select(a['qc_df'], cohort)
    keys = set(zip(q['group'].astype(str), q['dataset_id'].astype(str)))
    segmented = {g:{d:p for d,p in ds.items() if (str(g),str(d)) in keys} for g,ds in a['segmented'].items()}
    return segmented, q, select(a['response_df'], cohort)

def timecourses(analysis: dict, cohort: str, category_mode='all', bin_seconds=30) -> tuple[pd.DataFrame,pd.DataFrame]:
    segmented, q, _ = selected_view(analysis, cohort)
    animal = build_animal_timecourses(segmented, q, category_mode)
    if animal.empty:
        return animal, animal
    # Shared app functions preserve its segment/bin convention; no gap interpolation.
    binned = bin_timecourses(animal, bin_seconds, analysis['settings']['sampling_rate'])
    return binned, summarize_timecourses(binned, analysis['settings']['sampling_rate'], 5.0)

def ci(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Wald uncertainty intervals without changing source estimates or tests."""
    out = frame.copy()
    estimate = 'Estimate' if 'Estimate' in out else None
    if estimate and 'SE' in out:
        out['95% CI lower'] = out[estimate] - stats.norm.ppf(.975)*out['SE']
        out['95% CI upper'] = out[estimate] + stats.norm.ppf(.975)*out['SE']
        out['Finite estimate and SE'] = np.isfinite(out[estimate]) & np.isfinite(out['SE']) & out['SE'].gt(0)
    return out

def descriptive(values: pd.DataFrame, keys: list[str], outcome='z_response') -> pd.DataFrame:
    out = values.groupby(keys,observed=True)[outcome].agg(mean='mean',SD='std',n='count').reset_index()
    out['SEM'] = out['SD']/np.sqrt(out['n'])
    return out

def baseline_tests(values: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for compartment in ['dura','pia']:
        x=values.loc[values['compartment'].eq(compartment),'z_response'].dropna().to_numpy(float)
        if len(x)<2:
            continue
        t,p=stats.ttest_1samp(x,0)
        sem=stats.sem(x)
        bounds=stats.t.interval(.95,len(x)-1,loc=x.mean(),scale=sem)
        rows.append(dict(compartment=compartment,n=len(x),mean=x.mean(),SEM=sem,t=t,df=len(x)-1,
                         Raw_P=p,CI_lower=bounds[0],CI_upper=bounds[1]))
    out=pd.DataFrame(rows)
    if not out.empty:
        out['Holm_P']=holm_adjust(out.Raw_P.tolist()) if len(out)==2 else np.nan
    return out

def manuscript_statistics(analysis: dict) -> tuple[dict[str,pd.DataFrame],dict[str,str]]:
    """Run manuscript analyses with explicit cohorts, never global pooling switches."""
    tables, notes = {}, {}
    raw = analysis['response_df']
    for cohort,active in [('Levcromakalim','Levcro'),('Nitroglycerin','Nitroglycerin'),('PACAP 0.1 dosing','0_1mgkg_PACAP')]:
        w=response1(select(raw,cohort))
        if w.empty:
            notes[cohort]='Not calculated: no eligible data.'
            continue
        w['group']=w['group'].replace({'Vehicle':'Veh'})
        animal=session_values(w)
        animal['mouse_alias']=animal['mouse_alias'].map(canonical_mouse)
        mice=animal.groupby(['mouse_alias','group'],observed=True,as_index=False).agg(z_response=('z_response','mean'),sessions=('dataset_id','nunique'))
        tables[f'{cohort}_all_mouse_values']=mice
        tables[f'{cohort}_all_summary']=descriptive(mice,['group'])
        x=mice.loc[mice.group.eq(active),'z_response'].dropna()
        y=mice.loc[mice.group.eq('Veh'),'z_response'].dropna()
        if len(x) and len(y):
            if cohort=='Nitroglycerin':
                paired=mice.pivot(index='mouse_alias',columns='group',values='z_response')
                tables[f'{cohort}_paired_values']=paired.reset_index()
                if paired[[active,'Veh']].isna().any().any():
                    notes[f'{cohort}_test']='Not tested: fully paired manuscript design is not reproduced by loaded data.'
                else:
                    r=stats.wilcoxon(paired[active],paired.Veh,alternative='two-sided')
                    tables[f'{cohort}_all_test']=pd.DataFrame([{'Test':'Two-sided paired Wilcoxon','Statistic':r.statistic,'P':r.pvalue,'Pairs':len(paired)}])
            else:
                r=stats.mannwhitneyu(x,y,alternative='two-sided')
                tables[f'{cohort}_all_test']=pd.DataFrame([{'Test':'Two-sided Mann-Whitney U, active first','Statistic':r.statistic,'P':r.pvalue,'Active n':len(x),'Vehicle n':len(y)}])
        size=session_values(w,'vessel_size',['Small','Large'])
        omnibus,contrasts,note=mixed_model_size_interaction(size,'Veh',[active],holm_across_sizes=True)
        tables[f'{cohort}_size_input']=size
        tables[f'{cohort}_size_interaction']=omnibus
        tables[f'{cohort}_size_contrasts']=ci(contrasts)
        notes[f'{cohort}_size']=note
        # Supplementary descriptive Z and percentage-change summaries, not new inferential tests.
        for column,categories in [('category',['All vessels']),('vessel_size',['Small','Large']),('compartment',['dura','pia'])]:
            sw=w.copy()
            if column=='category': sw[column]='All vessels'
            sw=sw[sw[column].isin(categories)]
            for outcome in ['z_response','percent_change']:
                per_session=sw.groupby(['dataset_id','mouse_alias','group',column],observed=True,as_index=False)[outcome].mean()
                av=per_session.groupby(['mouse_alias','group',column],observed=True,as_index=False)[outcome].mean()
                tables[f'{cohort}_{column}_{outcome}_descriptive']=descriptive(av,['group',column],outcome)
        active_rows=w[w.group.eq(active)].copy()
        tables[f'{cohort}_vessel_response_input']=active_rows
        cls=np.select([active_rows.z_response.le(-1.96),active_rows.z_response.ge(1.96)],['Constricting','Dilating'],default='Neither threshold')
        counts=pd.Series(cls).value_counts().rename_axis('Class').reset_index(name='Vessels')
        counts['Percent']=100*counts.Vessels/len(active_rows)
        tables[f'{cohort}_response_classes_descriptive']=counts
    # Fig3h: pooled PACAP, original overlapping session replaced by dosing session.
    pooled=response1(select(analysis['pooled']['response_df'],'Fig3h pooled PACAP'))
    if not pooled.empty:
        vals,res,note=fit_pacap_compartment(pooled)
        tables['Fig3h_mouse_compartment_values']=vals
        tables['Fig3h_dura_pia_model']=ci(res)
        tables['Fig3h_baseline_tests']=baseline_tests(vals)
        tables['Fig3h_mouse_matched_for_Prism']=vals.pivot(index='mouse_alias',columns='compartment',values='z_response').reset_index()
        tables['Fig3h_vessel_input']=pooled
        notes['Fig3h']=note
    # Fig4/S3: original antibody-study sessions ONLY, not pooled with acute dosing.
    anti=response1(select(raw,'Anti-PACAP study'))
    if not anti.empty:
        av=session_values(anti)
        vals,res,diag,note=mixed_model_two_group_mouse_intercept(av,'PACAP','Anti-PACAP mAb')
        tables.update(Fig4_all_input=vals,Fig4_all_contrast=ci(res),Fig4_all_diagnostics=diag)
        notes['Fig4_all']=note
        for column,categories,label,prefix in [('compartment',['dura','pia'],'Vessel compartment','Fig4'),('vessel_size',['Small','Large'],'Vessel size','S3')]:
            av=session_values(anti,column,categories)
            vals,om,ct,diag,note=mixed_model_two_group_category_interaction(av,'PACAP','Anti-PACAP mAb',column,categories,label)
            for name,table in [('input',vals),('interaction',om),('contrasts',ci(ct)),('diagnostics',diag)]:
                tables[f'{prefix}_{column}_{name}']=table
            notes[f'{prefix}_{column}']=note
    tables['extreme_z_exclusions']=analysis['extreme_excluded']
    return tables,notes

def dosing_comparison(response: pd.DataFrame) -> dict[str,pd.DataFrame]:
    """Supplement S4b: independent acute doses; Bonferroni-adjusted Dunn tests."""
    w=response1(with_provenance(response))
    w=w[w.original_study.eq(DOSING)&w.original_treatment.isin(['Veh','Vehicle','0_1mgkg_PACAP','0_3mgkg_PACAP'])].copy()
    w['group']=w.original_treatment.replace({'Vehicle':'Veh'})
    vals=session_values(w)
    vals=vals.groupby(['mouse_alias','group'],observed=True,as_index=False).z_response.mean()
    groups=['Veh','0_1mgkg_PACAP','0_3mgkg_PACAP']
    if not all(g in set(vals.group) for g in groups):
        return {'S4b_values':vals}
    arrays=[vals.loc[vals.group.eq(g),'z_response'].to_numpy(float) for g in groups]
    kw=stats.kruskal(*arrays)
    joined=np.concatenate(arrays); ranks=stats.rankdata(joined); n=len(joined)
    _,tiecounts=np.unique(joined,return_counts=True)
    variance=n*(n+1)/12-np.sum(tiecounts**3-tiecounts)/(12*(n-1))
    starts=np.cumsum([0]+[len(x) for x in arrays]); rows=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        z=(ranks[starts[i]:starts[i+1]].mean()-ranks[starts[j]:starts[j+1]].mean())/np.sqrt(variance*(1/len(arrays[i])+1/len(arrays[j])))
        rows.append({'Comparison':f'{groups[i]} minus {groups[j]}','z':z,'Raw P':2*stats.norm.sf(abs(z))})
    dunn=pd.DataFrame(rows)
    dunn['Bonferroni P']=np.minimum(1,3*dunn['Raw P'])
    return {'S4b_values':vals,'S4b_Kruskal_Wallis':pd.DataFrame([{'H':kw.statistic,'P':kw.pvalue}]),'S4b_Dunn_pairwise':dunn}

def archive(tables: dict[str,pd.DataFrame], notes: dict, settings: dict) -> bytes:
    """Results archive is generated only on demand; it is not bundled with code."""
    buffer=BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,table in tables.items():
            name=''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
            z.writestr(name+'.csv',table.to_csv(index=False))
        record={'settings':settings,'notes':notes,'python':platform.python_version(),
                'packages':{p:version(p) for p in ['numpy','pandas','scipy','statsmodels','plotly']}}
        z.writestr('run_record.json',json.dumps(record,indent=2,default=str))
    return buffer.getvalue()
