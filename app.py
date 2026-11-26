"""HCCP II / HR Decision Analytics - public, precomputed-results-only Streamlit app.

The UI contains NO individual worker records or employer identifiers. All statistical
results are loaded from the previous project's fixed aggregate and model outputs.
The app does not fit models, simulate personnel-level risk, or imply causal effects.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from labels import labels, metrics, items, SEM_HYPOTHESES
from research_story import PROJECT_TITLE, MAIN_QUESTION
import viz

BASE = Path(__file__).resolve().parent
PUBLIC_FILES = ('public_summary.json','summary.json','sem_assessment.json','sem_structural.json',
                'ordinal_robustness.json','ordinal_satisfaction.json','priority_validation_v3.json',
                'validation.json','coefficients.csv','observed_relationships.csv','group_means_public.csv')
PRIVATE_FILES = ('data.csv','dashboard_preview.html','priority_inference.csv','research_header.html',
                 'sem_structural_bootstrap.csv','ordinal_thresholds.csv')
BLUE, TEAL, PURPLE, CORAL = '#2563EB','#0D9488','#7C3AED','#DC2626'

st.set_page_config(page_title='HR Decision Analytics | HCCPⅡ', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
st.markdown('<style>'+(BASE/'ui.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_results():
    def read_json(name):
        return json.loads((BASE/name).read_text(encoding='utf-8'))
    result = {key.removesuffix('.json'):read_json(key) for key in PUBLIC_FILES if key.endswith('.json')}
    result['coefficients'] = pd.read_csv(BASE/'coefficients.csv')
    result['observed'] = pd.read_csv(BASE/'observed_relationships.csv')
    result['groups'] = pd.read_csv(BASE/'group_means_public.csv')
    return result


def validate_public_bundle(r):
    p,s,sem = r['public_summary'],r['summary'],r['sem_structural']
    assert p['source_rows']==10759 and p['analysis_rows']==s['rows']==sem['n']==10731
    assert p['analysis_firms']==s['companies']==sem['firms']==497
    assert p['excluded_rows']==28
    assert not any((BASE/private).exists() for private in PRIVATE_FILES)
    assert set(r['coefficients']['model']).issuperset({'전체 통제','직무만족 전체 통제'})
    groups = r['groups']
    assert (groups['n']>=100).all() and (groups['firms_in_group']>=15).all()
    # No employee-level fields such as individual IDs, company IDs, or rows of answers.
    for f in ('coefficients.csv','group_means_public.csv','observed_relationships.csv'):
        assert 'W23ID1' not in (BASE/f).read_text(encoding='utf-8-sig').split('\n',1)[0]


R = load_results()
validate_public_bundle(R)
P,S,V,SEM,SA,OR,OS,PR = (
    R['public_summary'],R['summary'],R['validation'],R['sem_structural'],
    R['sem_assessment'],R['ordinal_robustness'],R['ordinal_satisfaction'],R['priority_validation_v3']
)
CO,OBS,G = R['coefficients'],R['observed'],R['groups']
SEM_EFFECTS={row['key']:row for row in SEM['effects']}

NAV_GROUPS = [
 ('연구 설계',[('overview','연구 개요'),('theory','이론 · 연구가설')]),
 ('데이터 분석',[('reliability','01  측정 신뢰도'),('descriptive','02  기술통계 · 집단 비교'),
                ('ols','03  다중회귀 OLS'),('ordinal','04  순서형 로짓'),('sem','05  구조방정식 SEM')]),
 ('의사결정',[('decision','HR 의사결정')]),
]
NAV_ITEMS=[item for _,group in NAV_GROUPS for item in group]
NAV_LABELS=dict(NAV_ITEMS)
ALL_NAV_KEYS=[key for key,_ in NAV_ITEMS]
if st.session_state.get('page') not in NAV_LABELS:
    st.session_state['page']='overview'


def move_to(key):
    st.session_state['page']=key


def eyebrow(value):
    st.markdown(f'<div class="overline">{escape(value)}</div>',unsafe_allow_html=True)


def page_intro(kicker,title,description):
    st.markdown(
        f'<div class="page-intro"><div class="overline">{escape(kicker)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def panel(title,body,kind='neutral',foot=''):
    css = 'decision-panel '+(kind if kind in ('blue','teal','amber','purple','neutral') else 'neutral')
    st.markdown(
        f'<div class="{css}"><div class="panel-title">{escape(title)}</div>'
        f'<div class="panel-body">{escape(body)}</div>'
        + (f'<div class="panel-foot">{escape(foot)}</div>' if foot else '')+'</div>',
        unsafe_allow_html=True,
    )


def evidence_triptych(question,interpretation,action,caution=None):
    cols=st.columns(3,gap='medium')
    with cols[0]: panel('분석 질문',question,'blue')
    with cols[1]: panel('결과 해석',interpretation,'teal')
    with cols[2]: panel('HR 의사결정',action,'purple')
    if caution:
        st.caption('해석 범위  |  '+caution)


def section_title(title,note=None):
    st.markdown(f'<div class="section-heading">{escape(title)}</div>',unsafe_allow_html=True)
    if note:
        st.caption(note)


def plot(fig,key):
    fig.update_layout(
        font=dict(family='Inter, Noto Sans KR, Malgun Gothic, sans-serif',size=12,color='#334155'),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20,r=34,t=38,b=35),
        hoverlabel=dict(bgcolor='white',font_color='#1F2937'),
    )
    fig.update_xaxes(gridcolor='#E9EDF3',zeroline=False)
    fig.update_yaxes(gridcolor='#E9EDF3',zeroline=False)
    st.plotly_chart(fig,use_container_width=True,key=key,config={'displaylogo':False})


def fmt(v):
    return f'{float(v):+.3f}'


def coef(model,variable):
    table=CO[(CO['model']==model)&(CO['variable']==variable)]
    if len(table)!=1:
        raise ValueError(f'Missing or repeated model coefficient: {model}/{variable}')
    return table.iloc[0]


HYPOTHESES=[
    ('H1','조직신뢰 → 직무만족','직무만족 전체 통제','trust','+'),
    ('H2','의사소통 → 직무만족','직무만족 전체 통제','comm','+'),
    ('H3','조직신뢰 → 이직 고려 의향','전체 통제','trust','−'),
    ('H4','의사소통 → 이직 고려 의향','전체 통제','comm','−'),
    ('H5','직무만족 → 이직 고려 의향','전체 통제','satisfaction','−'),
]


def ols_table():
    rows=[]
    for name,path,model,variable,expected in HYPOTHESES:
        e=coef(model,variable)
        b=float(e['coef'])
        actual='+' if b>0 else '−'
        p=float(e['p'])
        rows.append({
            '가설':name,'경로':path,'기대':expected,
            'OLS B':round(b,3), '95% 신뢰구간':f'[{fmt(e["ci_low"])}, {fmt(e["ci_high"])}]',
            'p': '<0.001' if p<.001 else f'{p:.3f}',
            '판정':'예상 방향 일치' if actual==expected else '예상과 반대',
        })
    return pd.DataFrame(rows)


def source_line():
    st.markdown(
      '<div class="data-source"><strong>DATA SOURCE</strong> · '
      '한국직업능력연구원(KRIVET) · HCCPⅡ 제4차(2023) 근로자 조사 '
      '· <a href="https://www.krivet.re.kr/kor/sub.do?menuSn=18" target="_blank">공식 조사 안내</a>'
      ' · <span>통계값은 연구자가 산출한 추정치</span></div>',
      unsafe_allow_html=True,
    )


with st.sidebar:
    # Render the entire identity + dataset banner in ONE measured Streamlit block.
    # Previously they were in two separate markdown blocks; overly aggressive CSS
    # collapsed their spacing and made the second line overlap the first.
    st.markdown(
      '<div class="sidebar-header">'
      '<div class="sidebar-brand"><span class="brand-symbol">PA</span>'
      '<div class="brand-text"><div class="brand-title">PEOPLE ANALYTICS</div>'
      '<div class="brand-subtitle">HR Decision Dashboard</div></div></div>'
      '<div class="sidebar-caption">HCCPⅡ 4차(2023) · 근로자 조사</div>'
      '</div>',
      unsafe_allow_html=True,
    )
    for group,items_in_group in NAV_GROUPS:
        # Native Streamlit caption reserves its own height and wraps predictably.
        st.caption(group)
        for item_key, item_label in items_in_group:
            is_active=(st.session_state['page']==item_key)
            st.button(item_label,key='nav_'+item_key,type='primary' if is_active else 'secondary',
                      use_container_width=True,on_click=move_to,args=(item_key,))
    st.markdown(
      '<div class="sidebar-footer"><div class="sidebar-meta">'
      '<strong>10,731명 · 497개 기업</strong><br>'
      '개별 응답 · 기업 식별정보 없음</div>'
      '<div class="sidebar-source">자료: KRIVET · HCCPⅡ 4차(2023)</div>'
      '<a href="https://www.krivet.re.kr/kor/sub.do?menuSn=18" '
      'target="_blank" rel="noopener noreferrer">공식 자료 보기 ↗</a>'
      '</div>',unsafe_allow_html=True,
    )

page=st.session_state['page']
st.markdown(
  '<div class="app-topbar"><span class="app-topbar-mark">HR</span>'
  '<strong>Decision Analytics</strong><span class="topbar-divider"></span>'
  f'<span>{escape(NAV_LABELS[page])}</span><span class="topbar-version">HCCPⅡ · 2023</span></div>',
  unsafe_allow_html=True,
)

if page=='overview':
    st.markdown(
      '<div class="executive-hero"><span class="executive-label">EXECUTIVE SUMMARY · 실제 조사자료 기반</span>'
      '<h1>조직의 신뢰와 소통,<br>어디부터 진단할 것인가?</h1>'
      '<p>HCCPⅡ 근로자 조사로 조직신뢰·의사소통·직무만족·이직 고려 의향의 관련성을 검토하고, '
      'HR이 우선 진단할 집단과 실행 절차를 제안합니다.</p>'
      '<div class="hero-tags"><span>H1–H5 주 검증</span><span>H6 탐색적 SEM</span>'
      '<span>비인과적 관찰연구</span></div></div>',unsafe_allow_html=True,
    )
    a,b,c,d=st.columns(4,gap='medium')
    a.metric('분석 근로자',f'{P["analysis_rows"]:,}명')
    b.metric('대상 기업',f'{P["analysis_firms"]:,}개')
    c.metric('주 검증 가설','H1–H5')
    d.metric('조사 기준 연도','2023')
    st.caption('전체 응답자 10,759명 중 연령 결측 28명 제외 · 공개 범위: 집계 통계 및 사전 추정 결과(원자료 제외)')
    section_title('핵심 연구 결과','표본 10,731명에 대한 조건부 관련성 결과입니다. 정책 효과나 인과관계를 추정하지 않았습니다.')
    cols=st.columns(3,gap='medium')
    with cols[0]: panel('01 · 이직의도 관련 요인','조직신뢰와 직무만족은 통제 후 이직 고려 의향과 음(-)의 관련성을 보였습니다.','blue','H3·H5 · OLS 주 분석')
    with cols[1]: panel('02 · 의사소통 해석 주의','H2는 OLS와 SEM에서 부호가 다르고, H4는 예상과 달리 양(+)의 부호를 보였습니다.','amber','측정 중첩·역인과·누락변수 가능성')
    with cols[2]: panel('03 · 실무 의사결정','집단을 먼저 진단한 뒤 현장 맥락을 확인하고, 효과는 별도의 시범사업에서 평가하도록 제안합니다.','teal','개인 위험점수·자동 인사결정 금지')
    left,right=st.columns([1.12,1],gap='large')
    with left:
        section_title('4개 핵심 지표의 관찰 평균','모두 5점 척도의 연구자 산출 평균. 응답 점수의 인과적 의미는 없습니다.')
        fig=go.Figure(go.Bar(
          x=list(P['overall_means'].values()),y=list(metrics.values()),orientation='h',
          marker_color=[viz.METRIC_COLORS[k] for k in metrics],
          text=[f'{P["overall_means"][k]:.2f}' for k in metrics],textposition='outside',
          hovertemplate='%{y}<br>관찰 평균 %{x:.3f}/5<extra></extra>',
        ))
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(range=[0,5.4],title='전체 표본 평균(5점 척도)')
        plot(viz.frame(fig,height=325),key='overview_means')
    with right:
        section_title('연구질문 → HR 판단','분석 결과가 실제 의사결정에 연결되는 경로')
        st.markdown(
            '<div class="step-list">'
            '<div><span>01</span><strong>실무질문과 이론</strong><small>어떤 조직요인이 직무만족·이직 고려와 관련되는가?</small></div>'
            '<div><span>02</span><strong>5단계 분석</strong><small>신뢰도 → 집단 평균 → OLS → 순서형 로짓 → SEM</small></div>'
            '<div><span>03</span><strong>HR 의사결정</strong><small>진단 후보 선별 → 현장 검증 → 제한적 시범사업 평가</small></div>'
            '</div>',unsafe_allow_html=True,
        )
    st.markdown('**실무질문** · '+MAIN_QUESTION)
    c1,c2,c3=st.columns(3)
    c1.button('이론 · 연구가설 보기',key='jump_theory',use_container_width=True,on_click=move_to,args=('theory',))
    c2.button('회귀분석 결과 보기',key='jump_ols',use_container_width=True,on_click=move_to,args=('ols',))
    c3.button('HR 의사결정 도구',key='jump_decision',use_container_width=True,on_click=move_to,args=('decision',))

    with st.expander('자료 출처 · 참고문헌 · AI 활용 · 연구윤리 상세 확인', expanded=False):
        st.markdown(
            '**조사 출처**: 한국직업능력연구원(KRIVET), 제2차 인적자본기업패널(HCCPⅡ) '
            '제4차(2023) 근로자 조사. 공식 전체 응답자 10,759명 중 연령 결측 '
            '28명을 제외한 10,731명(497개 기업)이 이 과제의 분석표본입니다. '
            '[공식 조사 안내](https://www.krivet.re.kr/kor/sub.do?menuSn=18) · '
            '[자료 생성·활용 보고서](https://www.krivet.re.kr/kor/sub.do?menuSn=12&pstNo=PB0000000425)'
        )
        st.caption('원자료·개인별 응답 행·기업 식별자는 공개 앱에 포함하지 않았습니다. '
                   '그러나 집계 결과의 공개 가능 범위는 원자료 이용약관을 확인해야 합니다.')
        st.markdown((BASE/'REFERENCES_AND_AI.md').read_text(encoding='utf-8'))
        st.warning('이 결과는 관찰자료에 기반한 집단 수준의 관련성으로, '
                   '직원 개인의 이직 위험 점수화나 자동 인사결정에 사용해서는 안 됩니다.')

elif page=='theory':
    page_intro('RESEARCH DESIGN  /  THEORY','이론적 배경 및 연구가설',
               '실무질문에서 X→Y의 예상 방향을 도출하고, 가설·통제변수·매개경로와 결과 해석을 연결합니다.')
    evidence_triptych('왜 신뢰·소통·만족을 측정하는가?',
        '사회교환이론·JD-R 모형·직무태도 선행연구에서 관계의 방향을 사전 설정합니다.',
        '이론과 다른 결과(H2·H4)를 숨기지 않고 추가 진단 과제로 연결합니다.',
        'HCCPⅡ 문항은 이론 전체를 직접 측정한 동일 척도가 아닙니다.')
    theory_lines=(BASE/'THEORY_AND_HYPOTHESES.md').read_text(encoding='utf-8').splitlines()
    # The page already has a title; omit the standalone Markdown document heading.
    if theory_lines and theory_lines[0].startswith('# '):
        theory_lines=theory_lines[1:]
    st.markdown('\n'.join(theory_lines).lstrip())
    section_title('H1–H6 · 예상 방향과 실제 결과','OLS는 H1–H5의 주 분석, SEM은 보완 분석이며 H6는 탐색적 간접경로입니다.')
    ols=ols_table()
    rows=[]
    for hyp in SEM_HYPOTHESES:
        key=hyp['id']; v=SEM_EFFECTS[key]
        o=ols.loc[ols['가설']==key]
        rows.append({'가설':key,'가설 경로':hyp['path'],'예상 방향':hyp['direction'],
                     'OLS B': '주 분석 아님' if o.empty else f'{o.iloc[0]["OLS B"]:+.3f}',
                     'SEM B':fmt(v['estimate']),
                     'SEM 95% CI':f'[{fmt(v["ci_low"])}, {fmt(v["ci_high"])}]',
                     '해석': ('탐색적 간접경로' if key=='H6' else '모형 간 부호 불일치' if key=='H2' else '예상과 반대' if key=='H4' else '방향 일치')})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('해석 유의사항 | OLS와 SEM은 추정·측정 방법이 달라 계수(B)의 크기를 직접 비교할 수 없습니다. H2의 부호 차이와 H4의 예상 반대 결과만으로 이론을 확증하거나 반증할 수 없습니다.')
    with st.expander('선행연구 원문 서지 및 DOI 확인'):
        references=(BASE/'REFERENCES_AND_AI.md').read_text(encoding='utf-8')
        st.markdown('### 참고문헌'+references.split('### 참고문헌',1)[1].split('### AI 활용 및 검증',1)[0])

elif page=='reliability':
    page_intro('DATA ANALYSIS  /  01','측정 신뢰도 및 타당성','조직신뢰·의사소통 문항의 내적 일관성을 살펴보고, 두 구성개념을 구분할 수 있는지 점검합니다.')
    c1,c2,c3=st.columns(3)
    c1.metric('조직신뢰 · Cronbach α',f'{P["alpha_trust"]:.3f}')
    c2.metric('의사소통 · Cronbach α',f'{P["alpha_comm"]:.3f}')
    c3.metric('2요인 CFA · 요인상관',f'{SA["latent_correlation"]:.3f}')
    section_title('문항별 5점 평균')
    pairs=[('조직신뢰',['W23Q22D','W23Q24D','W23Q24F'],BLUE),
           ('의사소통',['W23Q22F','W23Q24A','W23Q24B','W23Q24C'],TEAL)]
    left,right=st.columns(2,gap='large')
    for block,(title,codes,color) in zip((left,right),pairs):
        with block:
            st.markdown('**'+title+'**')
            fig=go.Figure(go.Bar(x=[P['item_means'][k] for k in codes],
               y=[items[k] for k in codes],orientation='h',marker_color=color,
               text=[f'{P["item_means"][k]:.2f}' for k in codes],textposition='outside',
               hovertemplate='%{y} · 평균 %{x:.3f}/5<extra></extra>'))
            fig.update_yaxes(autorange='reversed')
            fig.update_xaxes(range=[0,5.4],title='5점 평균')
            plot(viz.frame(fig,height=325),key='reliability_'+title)
    section_title('CFA 측정모형 적합도','5점 문항을 연속형으로 근사한 ML 적합 결과이며 기업 군집·순서형 분포는 보정하지 않았습니다.')
    cfa=pd.DataFrame([{'모형':name,'CFI':d['cfi'],'TLI':d['tli'],'RMSEA':d['rmsea'],'SRMR':d['srmr']}
       for name,d in [('1요인',SA['one_factor']),('2요인',SA['two_factor'])]])
    st.dataframe(cfa,use_container_width=True,hide_index=True)
    evidence_triptych('내적 일관성과 구성개념 구별은 같은가?',
        'α 값은 문항 내적 일관성을 보여주지만 2요인 CFA의 요인상관은 0.999로 구별이 불충분합니다.',
        '신뢰와 소통의 독립적 정책효과를 전제하지 말고, 문항 구성과 측정 타당성을 재검토해야 합니다.',
        '높은 신뢰도만으로 판별타당성이나 인과성을 확보했다고 주장할 수 없습니다.')

elif page=='descriptive':
    page_intro('DATA ANALYSIS  /  02','기술통계 및 집단 비교','사전 집계된 통계만 조회할 수 있으며, 개인 단위 필터·역추적 가능한 조합 조회는 제공하지 않습니다.')
    cols=st.columns(4)
    for (key,name),col in zip(metrics.items(),cols):
        col.metric(name,f'{P["overall_means"][key]:.2f} / 5')
    section_title('집단별 평균 비교','집단 분류와 관심 지표를 선택하면 공개 기준을 통과한 사전 집계값만 표시됩니다.')
    left,right=st.columns(2)
    dimension=left.selectbox('집단 분류',[k for k in labels if k in G['field'].unique()],format_func=lambda k:labels[k])
    metric=right.selectbox('분석 지표',list(metrics),format_func=lambda k:metrics[k])
    d=G[(G['field']==dimension)&(G['metric']==metric)].sort_values('group_mean')
    if len(d):
        fig=go.Figure(go.Bar(x=d['group_mean'],y=d['group_name'],orientation='h',
            marker_color=viz.METRIC_COLORS[metric],text=d['group_mean'].map(lambda v:f'{v:.2f}'),
            textposition='outside',customdata=d[['n','firms_in_group']].to_numpy(),
            hovertemplate='%{y}<br>평균 %{x:.3f}점<br>%{customdata[0]:,}명 / 기업 %{customdata[1]}개<extra></extra>'))
        fig.add_vline(x=float(P['overall_means'][metric]),line_dash='dash',line_color='#94A3B8',annotation_text='전체 평균')
        fig.update_xaxes(range=[0,5.4],title='5점 척도 평균')
        fig.update_yaxes(autorange='reversed')
        plot(viz.frame(fig,height=max(350,42*len(d)+110)),key='group_means_'+dimension+'_'+metric)
        show=d[['group_name','n','firms_in_group','group_mean']].rename(columns={'group_name':'집단','n':'응답자','firms_in_group':'기업 수','group_mean':'평균'})
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.download_button('집계 결과표 CSV 저장',show.to_csv(index=False).encode('utf-8-sig'),
                           file_name='hccp_group_summary.csv',mime='text/csv')
    else:
        st.info('해당 선택에서 공개 기준을 통과한 집계 행이 없습니다.')
    st.caption('공개 기준: 집단별 100명 이상·15개 기업 이상 및 비교집단에도 동일 기준을 적용한 사전 집계. 각 값은 반올림된 평균이며 원자료를 제공하지 않습니다.')
    section_title('핵심 변수의 관찰 상관','전체 공통표본의 단순 상관과 기업 단위 부트스트랩 95% 신뢰구간입니다.')
    plot(viz.observed_associations(OBS),key='observed_correlations')
    evidence_triptych('어떤 집단을 먼저 살펴볼 수 있는가?',
        '집단 평균 차이와 상관은 실제 관찰 자료의 분포와 함께 보는 탐색적 근거입니다.',
        '단순 평균 순위만으로 처방하지 않고, HR 의사결정 화면의 다중비교·집단규모 기준을 함께 확인합니다.',
        '집단 평균 차이는 개인의 특징·행동이나 정책 시행 효과를 뜻하지 않습니다.')

elif page=='ols':
    page_intro('DATA ANALYSIS  /  03','다중회귀분석 · H1–H5','기업 단위 군집강건 표준오차를 적용한 OLS 주 분석 결과를 조회합니다.')
    c1,c2,c3=st.columns(3)
    c1.metric('표본',f'{P["analysis_rows"]:,}명')
    c2.metric('기업 단위 군집',f'{P["analysis_firms"]:,}개')
    c3.metric('핵심 연구가설','H1–H5')
    section_title('가설별 회귀계수 및 신뢰구간','B는 다른 변수와 공변량을 통제한 조건부 관련성입니다.')
    st.dataframe(ols_table(),hide_index=True,use_container_width=True)
    st.markdown('**분석할 종속변수**')
    dv=st.radio('종속변수 선택',['직무만족','이직 고려 의향'],horizontal=True,label_visibility='collapsed')
    model='직무만족 전체 통제' if dv=='직무만족' else '전체 통제'
    plot(viz.regression_forest(CO.loc[CO['model']==model],{**metrics,**items}),key='ols_forest_'+model)
    st.caption(f'{model} 모형의 R² = {S["models"][model]["r2"]:.3f}. OLS 신뢰구간은 기업 단위 군집강건 표준오차에 기반합니다.')
    section_title('의사소통 계수 변화','모형에 추가되는 설명변수·통제변수에 따라 부호가 달라질 수 있습니다.')
    plot(viz.regression_change(CO),key='ols_coefficient_change')
    evidence_triptych('조직신뢰·소통은 만족과 이직 고려에 어떻게 관련되는가?',
        'H1·H3·H5는 기대 방향과 일치했습니다. H4는 예상과 반대로 양(+)이었습니다.',
        '신뢰·만족 관련 지표를 우선 모니터링하되, 소통의 방향성은 조직 맥락·역인과 가능성을 추가 조사합니다.',
        '높은 신뢰–소통 상관과 누락변수, 단면 측정 때문에 정책 인과효과를 주장할 수 없습니다.')
    with st.expander('통제변수와 추정범위'):
        st.markdown('성별·연령·학력·근속기간·직급·직무·고용형태·근로시간·산업·기업규모 등을 통제한 완전사례 10,731명을 분석했습니다. 통제가 모든 교란요인을 제거하거나 실제 이직을 예측하는 것은 아닙니다.')

elif page=='ordinal':
    page_intro('DATA ANALYSIS  /  04','순서형 로짓 · 강건성 점검','5점 서열 응답을 고려하여 OLS와 비교하고, 비례오즈 가정 검토 결과를 함께 제시합니다.')
    section_title('이직 고려 의향 · 순서형 로짓','OR>1은 더 높은 이직 고려 응답범주의 누적 오즈와 양(+)의 관련성입니다.')
    table=pd.DataFrame([{'요인':e['label'],'OLS B':fmt(e['ols_coef']),
      '순서형 로짓 B':fmt(e['ordinal_coef']),'OR':round(e['odds_ratio'],3),
      'OR 95% CI':f'[{e["or_ci_low"]:.3f}, {e["or_ci_high"]:.3f}]'} for e in OR['findings']])
    st.dataframe(table,hide_index=True,use_container_width=True)
    fig=go.Figure(go.Scatter(x=[e['odds_ratio'] for e in OR['findings']],
        y=[e['label'] for e in OR['findings']],mode='markers',marker=dict(size=12,color=BLUE),
        error_x=dict(type='data',symmetric=False,
          array=[e['or_ci_high']-e['odds_ratio'] for e in OR['findings']],
          arrayminus=[e['odds_ratio']-e['or_ci_low'] for e in OR['findings']],color='#94A3B8'),
        hovertemplate='%{y}<br>누적 오즈비 %{x:.3f}<extra></extra>'))
    fig.add_vline(x=1,line_dash='dash',line_color='#94A3B8')
    fig.update_xaxes(title='누적 오즈비 OR (95% 군집강건 CI)')
    plot(viz.frame(fig,height=310),key='ordinal_or_forest')
    section_title('직무만족 · 순서형 로짓','직무만족을 결과변수로 하는 보완적 결과로, 비례오즈 가정에 대한 별도 검정은 수행되지 않았습니다.')
    st.dataframe(pd.DataFrame([{'요인':e['label'],'OLS B':fmt(e['ols_coef']),
       '순서형 로짓 B':fmt(e['ordinal_coef']),'OR':round(e['odds_ratio'],3),
       'OR 95% CI':f'[{e["or_ci_low"]:.3f}, {e["or_ci_high"]:.3f}]'} for e in OS['findings']]),hide_index=True,use_container_width=True)
    section_title('비례오즈 가정 검토','초점 3개 변수의 경계별 계수 동일성 Wald 점검. 전체 통제변수의 전역 가정 검증은 아닙니다.')
    w=OR['threshold_analysis']['focus_slope_equality_wald']
    diag=pd.DataFrame([{'변수':row['label'],'Wald χ²':round(row['wald_chi2'],3),'df':row['df'],
      'p':'<0.001' if row['p']<.001 else f'{row["p"]:.3f}',
      '해석':'경계별 동일성 위배 가능성' if row['p']<.05 else '위배 근거가 확인되지 않음'} for row in w['per_factor']])
    st.dataframe(diag,hide_index=True,use_container_width=True)
    evidence_triptych('서열형 모형에서도 관계 방향이 유지되는가?',
        '이직 고려 모형에서 신뢰·만족은 OR<1, 소통은 OR>1로 방향이 유지되지만 소통 결과는 경계적입니다.',
        '모형 간 방향 일치만 확인하고 구성원별 위험 확률을 계산하거나 자동 인사결정에 사용하지 않습니다.',
        '직무만족의 경계별 동일성 위배 신호와 직무만족 모형의 가정 미검증을 명시해야 합니다.')

elif page=='sem':
    page_intro('DATA ANALYSIS  /  05','구조방정식 · SEM','H1–H5의 보완적 구조경로와 탐색적 간접경로 H6를 실제 저장된 추정값으로 보여줍니다.')
    section_title('SEM 적합 구조경로','조직신뢰는 3문항 잠재요인, 의사소통은 4문항 평균 관측 합성지표로 처리했습니다.')
    plot(viz.sem_fitted_paths(SEM),key='sem_structure')
    section_title('SEM 추정값 · 기업 부트스트랩 95% 구간')
    st.dataframe(pd.DataFrame([{'가설':h['id'],'경로':h['path'],'예상':h['direction'],
       'SEM B':fmt(SEM_EFFECTS[h['id']]['estimate']),
       '95% CI':f'[{fmt(SEM_EFFECTS[h["id"]]["ci_low"])}, {fmt(SEM_EFFECTS[h["id"]]["ci_high"])}]',
       '판정':SEM_EFFECTS[h['id']]['classification']} for h in SEM_HYPOTHESES]),
       hide_index=True,use_container_width=True)
    fit=SEM['fit']; cols=st.columns(4)
    for col,k in zip(cols,['cfi','tli','rmsea','srmr']):
        col.metric(k.upper(),f'{fit[k]:.3f}')
    section_title('직접·간접·전체 경로의 구분','H6는 모형에서 추정한 통계적 간접경로이며 시간적 매개효과를 입증하지 않습니다.')
    paths=[('직접경로 H3',SEM_EFFECTS['H3']),('탐색적 간접 H6',SEM_EFFECTS['H6']),('전체 경로 합',SEM_EFFECTS['trust_total'])]
    st.dataframe(pd.DataFrame([{'구분':name,'SEM B':fmt(e['estimate']),
        '기업 부트스트랩 95% CI':f'[{fmt(e["ci_low"])}, {fmt(e["ci_high"])}]'}
        for name,e in paths]),hide_index=True,use_container_width=True)
    st.caption('각 CI는 기업 500회 재표집 결과에서 별도로 계산한 값입니다. 신뢰구간들을 단순히 더한 값이 아닙니다.')
    evidence_triptych('조직신뢰–만족–이직 고려의 간접경로는 관찰되는가?',
        f'H6 통계적 간접경로는 {fmt(SEM_EFFECTS["H6"]["estimate"])}이며 구간에 0을 포함하지 않습니다.',
        '직무만족 경로를 추적하는 추가 연구를 고려하되 조직개입 효과를 확정하지 않습니다.',
        'SEM RMSEA 0.090, CFA 판별타당성 문제, 의사소통 H2 부호 차이 및 단면 자료의 한계가 있습니다.')

elif page=='decision':
    page_intro('HR DECISION SUPPORT  /  ACTION','HR 의사결정 지원','HR·현업 리더·경영진이 저장된 분석 근거를 확인하고, 실제 개입 전에 수행할 검증 절차를 정리합니다.')
    st.caption('활용 범위 | 저장된 통계 결과를 바탕으로 집단 수준의 HR 진단을 지원합니다. 개인별 이직 예측, 자동 인사 선별 및 정책 효과 시뮬레이션은 지원하지 않습니다.')
    section_title('담당자와 관심 지표 선택','선택에 따라 조회되는 실무 절차와 분석 근거가 달라집니다. 통계값은 새로 계산되지 않습니다.')
    a,b=st.columns(2)
    role=a.selectbox('의사결정 담당자',['HR 분석·기획','현업 리더','경영진·HR 의사결정자'])
    factor=b.selectbox('관심 조직지표',list(metrics),format_func=lambda key:metrics[key])
    ROLES={
      'HR 분석·기획':('집단 수준의 진단 후보를 선별하고 통계적 불확실성과 공정성 위험을 점검합니다.',
                   '표본·기업 수, 신뢰구간, BH 다중비교 보정, 공개 기준을 먼저 확인'),
      '현업 리더':('익명 의견 수렴과 업무 흐름 점검으로 통계적 차이의 현장 맥락을 확인합니다.',
                 '관리자 면담에서 개인을 특정하거나 불이익을 주지 않는 참여 절차 마련'),
      '경영진·HR 의사결정자':('진단에서 확인된 문제에 한해 제한된 시범사업과 평가 예산을 결정합니다.',
                           '목표지표, 예산, 기간, 사전·사후 및 가능하면 비교집단 설계 승인'),
    }
    FACTORS={
      'trust': ('조직신뢰',f'H1 OLS B={fmt(coef("직무만족 전체 통제","trust")["coef"])}; H3 OLS B={fmt(coef("전체 통제","trust")["coef"])}',
                '신뢰는 만족·이직 고려와 관련되지만 이를 높이는 정책의 효과는 미추정'),
      'comm': ('의사소통',f'H2 OLS B={fmt(coef("직무만족 전체 통제","comm")["coef"])}; H4 OLS B={fmt(coef("전체 통제","comm")["coef"])}',
                'H2는 SEM과 부호가 다르고 H4는 예상 반대이므로 단독 처방을 피함'),
      'satisfaction': ('직무만족',f'H5 OLS B={fmt(coef("전체 통제","satisfaction")["coef"])}',
                '직무만족은 이직 고려 의향과 음(-)의 관련성이 있으나 실제 퇴사를 예측하지 않음'),
      'turnover': ('이직 고려 의향','5점 단일문항 · 결과변수, 실제 이직률 아님',
                '집단 수준의 반복조사로 추세 확인; 개인별 위험점수나 불이익 조치 금지'),
    }
    title,evidence,caveat=FACTORS[factor]
    action,gate=ROLES[role]
    col1,col2=st.columns(2,gap='medium')
    with col1: panel(role+' · 다음 행동',action,'blue',gate)
    with col2: panel(title+' · 분석 근거',evidence,'teal',caveat)
    section_title('우선 진단 후보 · 사전 계산된 152개 비교','기본 기준: 불리한 차이 ≥0.3점, 응답자 50명·기업 10개 이상, CI 하한>0, BH q<0.05.')
    selected=PR['selected_at_default']
    st.dataframe(pd.DataFrame([{'집단':row['group_name'],'분류':row['dimension'],'지표':row['metric_name'],
       '응답자':row['n'],'기업':row['firms_in_group'],
       '불리한 평균 차이':round(row['adverse_difference'],3),
       '95% CI':f'[{row["adverse_ci_low"]:+.3f}, {row["adverse_ci_high"]:+.3f}]',
       'BH 보정 q':round(row['q_bh_152'],4)} for row in selected]),hide_index=True,use_container_width=True)
    if selected:
        lead=selected[0]
        st.caption(f'기본 기준 충족: {len(selected)}건. 예: {lead["dimension"]}–{lead["group_name"]}의 {lead["metric_name"]} 불리한 평균 차이 {lead["adverse_difference"]:+.3f}점. 현장 확인 후보이지 투자·개입 확정 대상이 아닙니다.')
    section_title('민감도 조회 · 기준을 달리하면 후보는 몇 개인가?','미리 계산된 조건별 통과 건수만 표시하며, 다른 조건에서 집단 이름을 추정하지 않습니다.')
    c1,c2=st.columns(2)
    cutoff=c1.select_slider('불리한 평균 차이 기준(점)',options=[0.2,0.3,0.4],value=0.3)
    minimum=c2.select_slider('집단 내 최소 응답자 수',options=[30,50,100],value=50)
    sensitivity=PR['sensitivity_counts'][f'g{cutoff:.1f}_n{minimum}']
    st.metric('해당 조합에서 선별되는 집단–지표 수',f'{int(sensitivity)}건')
    st.caption('위 민감도는 이전 분석환경에서 미리 계산되었습니다. 표본 공개 최소치(100명·15개 기업)는 이 차트와 다른 별도 기준입니다.')
    section_title('HR 실행 3단계')
    st.markdown(
      '<div class="step-list horizontal">'
      '<div><span>01</span><strong>진단 후보 선별</strong><small>집단 규모·CI·다중비교·차별 위험 검토</small></div>'
      '<div><span>02</span><strong>현장 원인 확인</strong><small>익명 의견 수렴·업무 프로세스 점검</small></div>'
      '<div><span>03</span><strong>시범사업 평가</strong><small>목표지표·기간·예산·사전/사후 평가</small></div>'
      '</div>',unsafe_allow_html=True,
    )
    evidence_triptych('어떤 부서에 인사 예산을 즉시 투입할 것인가?',
        '이 연구는 통계적 진단 후보를 제안하지만 인과 효과·정책 ROI는 추정하지 않았습니다.',
        '비용·실행 가능성·직원 보호 기준을 확인한 뒤 시범사업을 승인하고 효과는 별도로 평가합니다.',
        '개인별 위험 점수화, 자동 채용·평가·징계 판단에 사용해서는 안 됩니다.')

# Unified source footer on all eight sidebar pages.
st.divider()
source_line()
current_index=ALL_NAV_KEYS.index(page)
if current_index<len(ALL_NAV_KEYS)-1:
    next_page=ALL_NAV_KEYS[current_index+1]
    st.button('다음 화면 · '+NAV_LABELS[next_page],key='next_'+page,
              use_container_width=False,on_click=move_to,args=(next_page,))
st.caption('연구 용도: HR 집단 진단 및 후속 검증 보조 · 개인 단위 자동 인사결정/인과적 정책 효과 추정에 사용 불가')
