"""Local publication-analysis interface. Start: python -m streamlit run app.py"""
from dataclasses import asdict
import json
import pandas as pd
import streamlit as st
from core.pipeline import scan_data
from core.plots import timecourse_figure
from publication import Settings, COHORTS, prepare, timecourses, manuscript_statistics, dosing_comparison, inscopix_doses, archive

st.set_page_config(page_title='Vessel imaging publication analyses',layout='wide')
if st.session_state.get('publication_scope') != 'reviewer-v1':
    st.session_state.pop('statistics',None)
    st.session_state.publication_scope='reviewer-v1'
st.title('Vessel imaging publication analyses')
st.caption('Vessel-trace processing, time-course visualization and statistical methods. Input data are supplied locally.')
page=st.sidebar.radio('Page',['Data loading','Time courses','Manuscript statistics','Supplementary dosing'])

if page=='Data loading':
    root=st.text_input('Local study-data folder',placeholder='Choose the location of your approved input data')
    alias_upload=st.file_uploader('Optional treatment-label mapping (JSON)',type=['json'])
    c=st.columns(4)
    baseline=int(c[0].number_input('Baseline duration (s)',min_value=1,value=890))
    response=int(c[1].number_input('Response duration (s)',min_value=1,value=590))
    gap=float(c[2].number_input('New recording gap threshold (s)',min_value=.1,value=1.5))
    extreme=float(c[3].number_input('Extreme absolute-Z exclusion threshold',min_value=1.96,value=20.))
    st.info('Defaults: 890 baseline samples and 590 samples per response at 1 Hz. A vessel reaching |Z| ≥ the threshold at any frame is excluded from downstream analyses.')
    if st.button('Load and process local files',disabled=not root):
        try:
            with st.spinner('Processing accepted vessels and preparing explicit cohorts…'):
                aliases=json.load(alias_upload) if alias_upload is not None else None
                st.session_state.analysis=prepare(root,Settings(baseline_sec=baseline,response_sec=response,gap_threshold_sec=gap,extreme_z_threshold=extreme),treatment_aliases=aliases)
            st.session_state.pop('statistics',None)
            st.success('Processed. Your source files were not modified.')
        except Exception as exc:
            st.error(str(exc))
    if 'analysis' in st.session_state:
        a=st.session_state.analysis
        for name in ['loaded','issues','skipped','extreme_excluded']:
            st.subheader(name.replace('_',' ').title())
            st.dataframe(a[name],width='stretch')
        for key in ['scan_warnings','load_warnings','label_warnings','segment_warnings']:
            if a[key]: st.warning(str(a[key]))
else:
    if 'analysis' not in st.session_state:
        st.info('Start on Data loading. This distribution contains no input data.')
        st.stop()
    a=st.session_state.analysis
    if page=='Time courses':
        cohort=st.selectbox('Source cohort / analysis',COHORTS)
        mode=st.selectbox('Vessel grouping',['all','size','compartment'],format_func=lambda x:{'all':'All vessels','size':'Small / intermediate / large','compartment':'Dura / pia / unknown'}[x])
        bin_s=int(st.number_input('Plotting bin (s)',min_value=1,value=30))
        with st.spinner('Preparing time courses…'):
            animal,summary=timecourses(a,cohort,mode,bin_s)
        if summary.empty:
            st.warning('No eligible time-course data for this selection.')
        else:
            groups=st.multiselect('Treatments',list(summary.treatment.unique()),default=list(summary.treatment.unique()))
            categories=st.multiselect('Vessel categories',list(summary.category.unique()),default=list(summary.category.unique()))
            shown=summary[summary.treatment.isin(groups)&summary.category.isin(categories)]
            if mode=='all':
                st.plotly_chart(timecourse_figure(shown,'treatment',None,None,cohort,True),width='stretch')
            else:
                for treatment in groups:
                    st.plotly_chart(timecourse_figure(shown[shown.treatment.eq(treatment)],'category',None,None,f'{cohort}: {treatment}',True),width='stretch')
            st.caption('Mean ± SEM; each recording block is drawn separately, leaving pauses blank. Time-course bins are descriptive and are not tested individually. Downloads retain mouse and session identifiers for traceability.')
            st.download_button('Download time-course values (CSV ZIP)',archive({'timecourse_session_values':animal,'timecourse_summary':summary},{'cohort':cohort,'bin_seconds':bin_s},a['settings']),'timecourse_values.zip','application/zip')
    elif page=='Manuscript statistics':
        st.write('Treatment comparisons, vessel-size and vascular-compartment models, and comparisons with baseline.')
        st.caption('Additional baseline-diameter and diameter–response methods are available as functions in statistical_methods.py. ANALYSES.md maps the manuscript analyses to their implementations.')
        if st.button('Calculate manuscript statistics'):
            with st.spinner('Fitting manuscript models…'):
                st.session_state.statistics=manuscript_statistics(a)
        if 'statistics' in st.session_state:
            tables,notes=st.session_state.statistics
            st.download_button('Download all statistical results and exact model inputs',archive(tables,notes,a['settings']),'manuscript_statistics.zip','application/zip')
            name=st.selectbox('Result / model input',list(tables))
            st.dataframe(tables[name],width='stretch')
            with st.expander('Model descriptions and fit messages'):
                st.json(notes)
    else:
        st.subheader('Acute PACAP 0.1 / 0.3 mg/kg versus vehicle (S4b)')
        tables=dosing_comparison(a['response_df'])
        st.caption('Kruskal–Wallis test followed by two-sided, tie-corrected Dunn comparisons with Bonferroni adjustment across the three pairwise comparisons.')
        for name,table in tables.items():
            st.write(name); st.dataframe(table,width='stretch')
        st.subheader('Sequential Inscopix dosing (S4a)')
        st.write('Supply per-session dose times on the recorded-video clock. Full periods are bounded by the next dose. These dose summaries are descriptive; no inferential test is applied.')
        timing=st.file_uploader('Per-session dose timing CSV',type=['csv'])
        if timing is not None:
            try:
                tables.update(inscopix_doses(a,pd.read_csv(timing,dtype={'dataset_id':str})))
                for name,table in tables.items():
                    if name.startswith('S4a'): st.write(name); st.dataframe(table,width='stretch')
            except Exception as exc:
                st.error(str(exc))
        st.download_button('Download supplementary dosing tables',archive(tables,{},a['settings']),'supplementary_dosing.zip','application/zip')
