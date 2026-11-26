"""Basic offline public-bundle safety/data-consistency validation (stdlib only)."""
import csv
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
FORBIDDEN={
    'data.csv','dashboard_preview.html','research_header.html','priority_inference.csv',
    'sem_structural_bootstrap.csv','ordinal_thresholds.csv',
}
ALLOWED={
    'app.py','ui.css','README.md','run_local.ps1','preflight_public.py',
    'THEORY_AND_HYPOTHESES.md','REFERENCES_AND_AI.md','requirements.txt','labels.py',
    'viz.py','research_story.py','coefficients.csv','group_means_public.csv',
    'observed_relationships.csv','public_summary.json','summary.json','mediation.json',
    'sem_assessment.json','sem_structural.json','ordinal_robustness.json',
    'ordinal_satisfaction.json','priority_validation_v3.json','validation.json',
    '.gitignore','.streamlit/config.toml',
}
actual={x.relative_to(BASE).as_posix() for x in BASE.rglob('*') if x.is_file() and
        not any(part in ('__pycache__','.git','.pytest_cache') for part in x.relative_to(BASE).parts)}
forbidden=actual & FORBIDDEN
unexpected=actual-ALLOWED
assert not forbidden, f'Restricted files present: {sorted(forbidden)}'
assert not unexpected, f'Unexpected file(s): {sorted(unexpected)}'
with (BASE/'public_summary.json').open(encoding='utf-8') as file:
    source=json.load(file)
with (BASE/'summary.json').open(encoding='utf-8') as file:
    summary=json.load(file)
with (BASE/'sem_structural.json').open(encoding='utf-8') as file:
    sem=json.load(file)
assert (source['source_rows'],source['excluded_rows'],source['analysis_rows'],source['analysis_firms'])==(10759,28,10731,497)
assert source['analysis_rows']==summary['rows']==sem['n']
assert source['analysis_firms']==summary['companies']==sem['firms']
with (BASE/'group_means_public.csv').open(encoding='utf-8-sig',newline='') as file:
    for row in csv.DictReader(file):
        assert int(row['n'])>=100 and int(row['firms_in_group'])>=15
with (BASE/'coefficients.csv').open(encoding='utf-8-sig',newline='') as file:
    rows=list(csv.DictReader(file))
assert len(rows)>=5 and all(row['model'] and row['variable'] for row in rows)
assert all('data.csv' not in str(p) for p in actual)
print('PASS: files allowlist, minimum public aggregation thresholds, sample counts, modeled estimates.')
print('CAUTION: This check does not certify data-use/publication permission or live web-app uptime.')
