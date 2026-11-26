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
    if key != st.session_state.get('page'):
        st.session_state['page']=key
        st.session_state['_page_scroll_pending']=True
        st.session_state['_page_scroll_nonce']=st.session_state.get('_page_scroll_nonce',0)+1


def scroll_to_page_top():
    """Reset the main viewport only after an actual navigation event."""
    if not st.session_state.pop('_page_scroll_pending',False):
        return
    # The script is static except for an internally generated integer.
    # A new nonce ensures the one-pixel iframe executes on every navigation.
    nonce=int(st.session_state.get('_page_scroll_nonce',0))
    script="""
<script>
(() => {
    const win = window.parent;
    const doc = win.document;
    const reset = () => {
        const main = doc.querySelector('[data-testid="stMain"]')
                  || doc.querySelector('section.main')
                  || doc.querySelector('[data-testid="stAppViewContainer"]');
        if (main) main.scrollTop = 0;
        const root = doc.scrollingElement;
        if (root) root.scrollTop = 0;
        win.scrollTo(0, 0);
    };
    reset();
    requestAnimationFrame(() => requestAnimationFrame(reset));
    setTimeout(reset, 120);
})();
</script>
<!-- navigation: SCROLL_NONCE -->
""".replace('SCROLL_NONCE',str(nonce))
    if hasattr(st,'iframe'):
        st.iframe(script,height=1,tab_index=-1)
    else:
        import streamlit.components.v1 as components
        components.html(script,height=1,scrolling=False)


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


def source_line(plain=False):
    css='data-source overview-source' if plain else 'data-source'
    st.markdown(
      f'<div class="{css}"><strong>DATA SOURCE</strong> · '
      '한국직업능력연구원(KRIVET) · HCCPⅡ 제4차(2023) 근로자 조사 '
      ' · <span>통계값은 연구자가 산출한 추정치</span></div>',
      unsafe_allow_html=True,
    )


def overview_reference_footer():
    reference_text=(BASE/'REFERENCES_AND_AI.md').read_text(encoding='utf-8')
    reference_label='자료 출처 · 참고문헌 · AI 활용 · 연구윤리 ↗'
    dialog=getattr(st,'dialog',None) or getattr(st,'experimental_dialog',None)
    if dialog is not None:
        @dialog('연구 참고 정보',width='large')
        def show_reference_information():
            st.markdown(reference_text)
        # The container key scopes link styling to this footer control only.
        try:
            link_container=st.container(key='overview_reference_link')
        except TypeError:
            link_container=st.container()
        with link_container:
            if st.button(reference_label,key='overview_reference_info',use_container_width=False):
                show_reference_information()
    else:
        with st.expander('자료 출처 · 참고문헌 · AI 활용 · 연구윤리',expanded=False):
            st.markdown(reference_text)


with st.sidebar:
    # Render the entire identity + dataset banner in ONE measured Streamlit block.
    # Previously they were in two separate markdown blocks; overly aggressive CSS
    # collapsed their spacing and made the second line overlap the first.
    st.markdown(
      '<div class="sidebar-header">'
      '<div class="sidebar-brand"><span class="brand-symbol">PA</span>'
      '<div class="brand-text"><div class="brand-title">PEOPLE ANALYTICS</div>'
      '<div class="brand-subtitle">HR Decision Dashboard</div></div></div>'
      '<div class="sidebar-caption">HCCPⅡ 2023 · 실제 조사자료 기반</div>'
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
      '<div class="executive-hero"><span class="executive-label">EXECUTIVE SUMMARY</span>'
      '<h1>조직의 신뢰와 소통,<br>어디부터 진단할 것인가?</h1>'
      '<p>HCCPⅡ 근로자 조사로 조직신뢰·의사소통·직무만족·이직 고려 의향의 관련성을 검토하고, '
      'HR이 우선 진단할 집단과 실행 절차를 제안합니다.</p>'
      '<div class="hero-tags"><span>H1–H5 주 검증</span><span>H6 탐색적 SEM</span>'
      '<span>비인과적 관찰연구</span></div></div>',unsafe_allow_html=True,
    )
    st.markdown('**실무질문** · '+MAIN_QUESTION)

    a,b,c,d=st.columns(4,gap='medium')
    a.metric('분석 근로자',f'{P["analysis_rows"]:,}명')
    b.metric('대상 기업',f'{P["analysis_firms"]:,}개')
    c.metric('주 검증 가설','H1–H5')
    d.metric('조사 기준 연도','2023')
    st.caption('전체 응답자 10,759명 중 연령 결측 28명 제외 · 공개 범위: 집계 통계 및 사전 추정 결과(원자료 제외)')
    section_title('핵심 연구 결과','표본 10,731명에 대한 조건부 관련성 결과입니다. 정책 효과나 인과관계를 추정하지 않았습니다.')
    cols=st.columns(3,gap='medium')
    with cols[0]: panel('01 · 이직 고려 의향 관련 요인','조직신뢰와 직무만족은 통제 후 이직 고려 의향과 음(-)의 관련성을 보였습니다.','blue','H3·H5 · OLS 주 분석')
    with cols[1]: panel('02 · 의사소통 해석 주의','H2는 OLS와 SEM에서 부호가 다르고, H4는 예상과 달리 양(+)의 부호를 보였습니다.','amber','측정 중첩·역인과·누락변수 가능성')
    with cols[2]:
        candidates=PR['selected_at_default']
        if len(candidates)==1:
            lead=candidates[0]
            panel('03 · 실무 의사결정',
                  f'{lead["group_name"]} {lead["dimension"]}가 우선 진단 후보로 도출되었습니다. 현장 원인 확인 후 시범사업 평가를 제안합니다.',
                  'teal',f'{lead["metric_name"]} {lead["adverse_difference"]:+.3f}점 · {PR["contrasts"]}개 비교 중 1건')
        else:
            panel('03 · 실무 의사결정',
                  '통계적 선별 결과를 바탕으로 현장 원인을 확인하고, 효과는 별도의 시범사업에서 평가합니다.',
                  'teal',f'{PR["contrasts"]}개 비교 · 기본 기준 충족 {len(candidates)}건')
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
        section_title('연구 진행 흐름','분석 결과가 실제 HR 의사결정에 연결되는 경로')
        st.markdown(
            '<div class="step-list">'
            '<div><span>01</span><strong>실무질문과 이론</strong><small>어떤 조직요인이 직무만족·이직 고려와 관련되는가?</small></div>'
            '<div><span>02</span><strong>5단계 분석</strong><small>신뢰도 → 집단 평균 → OLS → 순서형 로짓 → SEM</small></div>'
            '<div><span>03</span><strong>HR 의사결정</strong><small>진단 후보 선별 → 현장 검증 → 제한적 시범사업 평가</small></div>'
            '</div>',unsafe_allow_html=True,
        )

elif page=='theory':
    page_intro('RESEARCH DESIGN  /  THEORY','이론적 배경 및 연구가설',
               '실무질문에서 X→Y의 예상 방향을 도출하고, 가설·통제변수·매개경로와 결과 해석을 연결합니다.')
    theory_lines=(BASE/'THEORY_AND_HYPOTHESES.md').read_text(encoding='utf-8').splitlines()
    # The page already has a title; omit the standalone Markdown document heading.
    if theory_lines and theory_lines[0].startswith('# '):
        theory_lines=theory_lines[1:]
    theory_text='\n'.join(theory_lines).lstrip()
    theory_design,discussion_heading,theory_discussion=theory_text.partition('### 3. 이론과 다르게 나타난 결과를 어떻게 해석하는가?')
    st.markdown(theory_design.rstrip())
    section_title('3. 연구가설과 분석 결과 비교','OLS는 H1–H5의 주 분석, 순서형 로짓은 보완 분석, SEM은 구조경로와 H6의 탐색적 간접경로 분석입니다.')
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
    st.caption('해석 유의사항 | OLS와 SEM은 추정·측정 방법이 달라 계수(B)의 크기를 직접 비교할 수 없습니다.')
    if discussion_heading:
        discussion='### 4. 이론적 해석 및 대안 설명'+theory_discussion
        discussion=discussion.replace('### 4. 대안 설명을 줄이기 위한 후속 연구 제안','### 5. 후속 연구 방향',1)
        st.markdown(discussion.strip())

elif page=='reliability':
    page_intro('DATA ANALYSIS  /  01','측정 신뢰도 및 타당성','조직신뢰·의사소통 문항의 내적 일관성을 살펴보고, 두 구성개념을 구분할 수 있는지 점검합니다.')
    c1,c2,c3=st.columns(3)
    c1.metric('조직신뢰 · Cronbach α',f'{P["alpha_trust"]:.3f}')
    c2.metric('의사소통 · Cronbach α',f'{P["alpha_comm"]:.3f}')
    c3.metric('2요인 CFA · 요인상관',f'{SA["latent_correlation"]:.3f}')
    panel('측정 타당성 · 판별타당성 미확보',
          f'두 요인 간 상관이 {SA["latent_correlation"]:.3f}이며, 2요인 CFA의 CFI {SA["two_factor"]["cfi"]:.3f}, RMSEA {SA["two_factor"]["rmsea"]:.3f}는 측정모형의 한계를 보여줍니다.',
          'amber','내적 일관성이 양호하더라도 조직신뢰와 의사소통을 통계적으로 명확하게 구분하기는 어렵습니다.')
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
    st.dataframe(cfa.round(3),use_container_width=True,hide_index=True)
    st.markdown('**측정 결과 해석** · 1요인·2요인 모형 모두 RMSEA가 높고, 2요인 모형의 적합도 개선이 확인되지 않았습니다.')
    st.caption('후속 검토 | 조직신뢰와 의사소통을 구분해 해석하기 전에 문항 구조와 순서형·기업 군집 보정을 추가로 검토할 필요가 있습니다.')

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
    st.markdown('**집단 비교의 활용** · 집단별 평균과 변수 간 상관은 현장 진단의 탐색적 근거입니다. 평균 순위만으로 판단하지 않고 HR 의사결정 페이지의 집단 규모·신뢰구간·다중비교 기준을 함께 확인합니다.')
    st.caption('집단 평균 차이를 개인의 특성이나 정책 시행 효과로 해석하지 않습니다.')

elif page=='ols':
    page_intro('DATA ANALYSIS  /  03','다중회귀분석 · H1–H5','개인·직무·기업 특성을 통제하고 기업 단위 군집강건 표준오차를 적용한 다중회귀분석 결과입니다.')
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
    panel('회귀분석 결과 해석',
          'OLS에서 H1·H2·H3·H5는 가설의 예상 방향과 일치했으며, H4는 예상과 반대로 양(+)의 관련성을 보였습니다.',
          'teal','신뢰·만족 관련 지표를 우선 모니터링하고, 소통의 방향성은 조직 맥락과 함께 검토합니다.')
    section_title('분석 가정 및 해석상의 한계','확인된 사항과 추가 검증이 필요한 항목을 구분하여 분석 결과의 해석 범위를 제시합니다.')
    scale_corr=V['sem_feasibility']['trust_comm_scale_corr']
    st.dataframe(pd.DataFrame([
        {'점검 항목':'기업 내 응답 상관·이분산','확인 상태':'추정 시 반영','확인 근거':f'{P["analysis_firms"]}개 기업 단위 군집강건 표준오차 적용','해석 및 후속 점검':'기업 내 상관과 이분산을 고려한 추론. 선형성이나 교란 문제를 해결하지는 않음'},
        {'점검 항목':'설명변수 중첩·다중공선성','확인 상태':'부분 점검','확인 근거':f'신뢰·소통 합성척도 상관 r={scale_corr:.3f}; 전체 VIF 미확인','해석 및 후속 점검':'개별 계수의 크기·부호에 주의. 원자료로 VIF 및 설계행렬 점검 필요'},
        {'점검 항목':'선형성·잔차 분포·영향점','확인 상태':'미확인','확인 근거':'정식 잔차 진단 결과가 저장되어 있지 않음','해석 및 후속 점검':'잔차-적합값, 잔차 분포 및 영향점 진단 필요'},
        {'점검 항목':'5점 응답의 순서척도 특성','확인 상태':'대안 모형 비교','확인 근거':'5점 서열 응답을 연속형으로 근사하여 분석','해석 및 후속 점검':'대안 모형의 비교 결과와 가정 검토는 04 순서형 로짓에서 확인'},
        {'점검 항목':'모형 사양 민감성','확인 상태':'부분 점검','확인 근거':'H4 단순·전체 통제 계수 부호 변화','해석 및 후속 점검':'통제변수 구성에 따른 계수 변화에 주의; 측정구조 비교는 05 SEM에서 확인'},
        {'점검 항목':'교란·시간적 선후·조사 가중치','확인 상태':'해석 제한','확인 근거':'관측된 공변량 통제; 2023년 단면자료, 가중치 미적용','해석 및 후속 점검':'미관측 교란과 대표성에 주의. 반복 조사 및 추가 검증 필요'},
    ]),hide_index=True,use_container_width=True)
    st.caption('검증 범위 | 군집강건 표준오차는 추론 방법이며 가정 검정 결과가 아닙니다. 위 표는 기존 점검 결과를 정리한 것으로, 추가 통계 진단을 수행한 것은 아닙니다.')

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
    section_title('이직 고려 의향 모형의 비례오즈 가정 검토','조직신뢰·의사소통·직무만족의 응답 경계별 계수 동일성을 Wald 검정으로 점검했습니다. 전체 변수에 대한 검정은 아닙니다.')
    w=OR['threshold_analysis']['focus_slope_equality_wald']
    diag=pd.DataFrame([{'변수':row['label'],'Wald χ²':round(row['wald_chi2'],3),'df':row['df'],
      'p':'<0.001' if row['p']<.001 else f'{row["p"]:.3f}',
      '해석':'경계별 동일성 위배 가능성' if row['p']<.05 else '위배 근거가 확인되지 않음'} for row in w['per_factor']])
    st.dataframe(diag,hide_index=True,use_container_width=True)
    st.markdown('**강건성 점검 결과** · 순서형 로짓에서도 H1–H5의 OLS 계수 방향이 일관되게 유지되었습니다. 다만 이직 고려 의향 모형의 비례오즈 가정에서 일부 위배 신호가 확인되었으므로, 결과의 방향적 일관성은 확인되었으나 모형의 강건성이 완전히 확보되었다고 단정할 수 없습니다.')
    st.caption('해석 유의사항 | 이직 고려 의향 모형에서 직무만족의 경계별 계수 동일성 위배 신호가 있으며, 의사소통 오즈비의 95% 신뢰구간 하한은 1에 가깝습니다. 직무만족을 결과변수로 하는 모형의 비례오즈 가정은 별도로 검증하지 않았습니다.')

elif page=='sem':
    page_intro('DATA ANALYSIS  /  05','구조방정식 · SEM','H1–H5의 보완적 구조경로와 탐색적 간접경로 H6를 실제 저장된 추정값으로 보여줍니다.')
    section_title('SEM 모형 적합도 및 해석 범위','탐색적 혼합형 SEM의 적합도를 여러 지표로 종합하여 판단합니다.')
    fit=SEM['fit']; cols=st.columns(4,gap='medium')
    for col,k in zip(cols,['cfi','tli','rmsea','srmr']):
        col.metric(k.upper(),f'{fit[k]:.3f}')
    a,b=st.columns(2,gap='medium')
    with a:
        panel('모형 적합도 · 일부 한계',
              f'CFI {fit["cfi"]:.3f}, TLI {fit["tli"]:.3f}, SRMR {fit["srmr"]:.3f}는 비교적 양호하나, RMSEA {fit["rmsea"]:.3f}은 일반적 참고 수준(0.08)을 초과합니다.',
              'amber','참고 수준은 절대적인 합격 기준이 아닙니다. 전체 모형의 적합성이 충분히 확보됐다고 단정하지 않습니다.')
    with b:
        panel('측정 타당성 · 해석 주의',
              f'2요인 CFA의 요인상관은 {SA["latent_correlation"]:.3f}이며, 혼합형 SEM의 조직신뢰·의사소통 상관도 {SEM_EFFECTS["trust_comm_latent_observed_corr"]["estimate"]:.3f}입니다.',
              'neutral','관측 합성지표를 사용해도 판별타당성 문제가 해결되는 것은 아닙니다. H2·H4 등 개별 경로의 해석에 주의합니다.')
    st.caption('추정 범위 | 5점 응답을 연속형으로 근사한 최대우도 추정입니다. 기업 단위 부트스트랩은 경로계수의 신뢰구간에 반영되며, 모형 적합도 지표 자체는 기업 군집 보정되지 않았습니다.')
    section_title('SEM 구조모형 및 경로도','조직신뢰는 3문항 잠재요인, 의사소통은 4문항 평균 관측 합성지표로 처리했습니다. 탐색적 구조경로이며 인과관계를 뜻하지 않습니다.')
    plot(viz.sem_fitted_paths(SEM),key='sem_structure')
    section_title('가설별 경로계수 및 신뢰구간','기업 단위 부트스트랩 95% 신뢰구간을 제시합니다.')
    st.dataframe(pd.DataFrame([{'가설':h['id'],'경로':h['path'],'예상':h['direction'],
       'SEM B':fmt(SEM_EFFECTS[h['id']]['estimate']),
       '95% CI':f'[{fmt(SEM_EFFECTS[h["id"]]["ci_low"])}, {fmt(SEM_EFFECTS[h["id"]]["ci_high"])}]',
       '판정':SEM_EFFECTS[h['id']]['classification']} for h in SEM_HYPOTHESES]),
       hide_index=True,use_container_width=True)
    section_title('직접·간접·전체 경로의 구분','H6는 모형에서 추정한 통계적 간접경로이며 시간적 매개효과를 입증하지 않습니다.')
    paths=[('직접경로 H3',SEM_EFFECTS['H3']),('탐색적 간접 H6',SEM_EFFECTS['H6']),('전체 경로 합',SEM_EFFECTS['trust_total'])]
    st.dataframe(pd.DataFrame([{'구분':name,'SEM B':fmt(e['estimate']),
        '기업 부트스트랩 95% CI':f'[{fmt(e["ci_low"])}, {fmt(e["ci_high"])}]'}
        for name,e in paths]),hide_index=True,use_container_width=True)
    st.caption('각 CI는 기업 500회 재표집 결과에서 별도로 계산한 값입니다. 신뢰구간들을 단순히 더한 값이 아닙니다.')
    st.markdown(f'**간접경로 해석** · H6의 통계적 간접경로는 {fmt(SEM_EFFECTS["H6"]["estimate"])}이며 기업 부트스트랩 95% 신뢰구간에 0을 포함하지 않습니다. 탐색적 관련성으로 해석합니다.')
    st.caption('후속 연구 | 반복 조사에서 조직신뢰와 직무만족, 이후 이직 고려 의향을 측정하여 시간적 선후를 검토합니다.')

elif page=='decision':
    page_intro('HR DECISION SUPPORT  /  ACTION','HR 의사결정 지원','HR 담당자, 현업 리더 및 경영진이 분석 결과를 확인하고, 실제 개선 조치를 시행하기 전에 필요한 검증 절차를 검토할 수 있습니다.')
    st.caption('활용 범위 | 집단 수준의 진단 근거와 후속 평가 절차를 제공합니다. 개인별 이직 예측, 자동 인사결정 및 정책 효과 시뮬레이션은 지원하지 않습니다.')
    section_title('우선 진단 후보 · 사전 계산된 152개 비교','기본 기준: 불리한 차이 ≥0.3점, 응답자 50명·기업 10개 이상, CI 하한>0, BH q<0.05.')
    selected=PR['selected_at_default']
    st.dataframe(pd.DataFrame([{'집단':row['group_name'],'분류':row['dimension'],'지표':row['metric_name'],
       '응답자':row['n'],'기업':row['firms_in_group'],
       '불리한 평균 차이':round(row['adverse_difference'],3),
       '95% CI':f'[{row["adverse_ci_low"]:+.3f}, {row["adverse_ci_high"]:+.3f}]',
       'BH 보정 q':round(row['q_bh_152'],4)} for row in selected]),hide_index=True,use_container_width=True)
    if selected:
        st.caption(f'기본 기준 충족: {len(selected)}건. 현장 확인 후보이며 투자·개입 확정 대상은 아닙니다.')
    followups=PR.get('adjusted_selected_follow_up',[])
    customer_support=next((row for row in selected
                           if row['dimension']=='직무' and row['group_name']=='대고객지원'
                           and row['metric_name']=='이직 고려 의향'),None)
    adjusted=next((row for row in followups
                   if row['field']=='W23Q04' and row['code']==5 and row['metric']=='turnover'),None)
    if customer_support is not None and adjusted is not None:
        section_title('우선 진단 후보의 추가 분석','대고객지원 직무의 이직 고려 의향 차이를 공변량 조정 전후로 확인합니다.')
        st.dataframe(pd.DataFrame([
            {'분석 구분':'집단 대 나머지 평균 차이',
             '추정값(점)':fmt(customer_support['adverse_difference']),
             '95% CI':f'[{fmt(customer_support["adverse_ci_low"])}, {fmt(customer_support["adverse_ci_high"])}]',
             '검정 근거':f'BH-FDR q={customer_support["q_bh_152"]:.4f}',
             '분석 목적':'기본 선별 기준에 따른 진단 후보 도출'},
            {'분석 구분':'공변량 조정 후 회귀계수',
             '추정값(점)':fmt(adjusted['adjusted_coef']),
             '95% CI':f'[{fmt(adjusted["adjusted_ci"][0])}, {fmt(adjusted["adjusted_ci"][1])}]',
             '검정 근거':f'p={adjusted["adjusted_p"]:.4f}',
             '분석 목적':'집단 구성·근무 여건을 고려한 탐색적 추가 검토'},
        ]),hide_index=True,use_container_width=True)
        panel('실무 판단 · 우선 현장 진단 근거',
              f'대고객지원 직무의 평균 차이 {fmt(customer_support["adverse_difference"])}점은 공변량 조정 후 {fmt(adjusted["adjusted_coef"])}점으로 작아졌으나 양(+)의 방향이 유지되고 신뢰구간에 0을 포함하지 않습니다.',
              'teal','현업 리더는 업무량·교대근무·고객응대 부담 및 조직 지원을 익명 조사로 확인하고, 확인된 원인에 한해 개선 과제를 검토합니다.')
        st.caption('해석 범위 | 공변량 조정은 후보 선별 후 탐색적 분석으로, p값을 다중검정·선별 보정된 확증적 근거로 해석하지 않습니다. 기본 0.3점 기준은 최초 평균 차이에만 적용합니다. 조정 전후 차이는 특정 원인이나 직무의 인과효과를 증명하지 않습니다.')
    section_title('민감도 조회 · 기준을 달리하면 후보는 몇 개인가?','미리 계산된 조건별 통과 건수만 표시하며, 다른 조건에서 집단 이름을 추정하지 않습니다.')
    c1,c2=st.columns(2)
    cutoff=c1.select_slider('불리한 평균 차이 기준(점)',options=[0.2,0.3,0.4],value=0.3)
    minimum=c2.select_slider('집단 내 최소 응답자 수',options=[30,50,100],value=50)
    sensitivity=PR['sensitivity_counts'][f'g{cutoff:.1f}_n{minimum}']
    st.metric('해당 조합에서 선별되는 집단–지표 수',f'{int(sensitivity)}건')
    st.caption('선택한 조건은 사전 계산된 후보 건수 조회에만 적용되며, 위 기본 기준의 후보표와 추가 분석 결과는 변경되지 않습니다. 공개 평균표 기준(100명·15개 기업)과 진단 후보 선별 기준은 별개입니다.')
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
    section_title('HR 실행 3단계')
    st.markdown(
      '<div class="step-list horizontal">'
      '<div><span>01</span><strong>진단 후보 선별</strong><small>집단 규모·CI·다중비교·차별 위험 검토</small></div>'
      '<div><span>02</span><strong>현장 원인 확인</strong><small>익명 의견 수렴·업무 프로세스 점검</small></div>'
      '<div><span>03</span><strong>시범사업 평가</strong><small>목표지표·기간·예산·사전/사후 평가</small></div>'
      '</div>',unsafe_allow_html=True,
    )
    section_title('후속 평가 지표 및 기대효과','현장 진단을 거쳐 개선 과제가 확인된 경우에 적용할 시범사업 평가 계획입니다.')
    st.dataframe(pd.DataFrame([
        {'구분':'핵심','평가 지표':'이직 고려 의향','확인 방법':'동일 5점 문항으로 사전·사후 설문','목표 방향':'감소'},
        {'구분':'핵심','평가 지표':'직무만족','확인 방법':'동일 5점 문항으로 반복 측정','목표 방향':'향상'},
        {'구분':'보조','평가 지표':'조직신뢰','확인 방법':'동일 3개 문항 및 익명 의견 수렴','목표 방향':'향상'},
        {'구분':'보조','평가 지표':'의사소통','확인 방법':'동일 4개 문항 및 업무 흐름 점검','목표 방향':'확인된 소통 문제 개선'},
        {'구분':'장기','평가 지표':'실제 자발적 이직률','확인 방법':'동일 기간 자발적 퇴사자 수 ÷ 평균 재직자 수','목표 방향':'감소'},
    ]),hide_index=True,use_container_width=True)
    st.caption('평가 계획 | HR은 시범사업 시작 전 기준값·평가 기간·성공 및 중단 기준을 정하고, 현업 리더는 익명 설문과 업무 맥락을 확인합니다. 가능하면 비교집단과 동일한 시점에 반복 측정하고, 경영진은 결과·비용·직원 보호 여부를 종합하여 확대 또는 중단을 판단합니다.')
    st.caption('기대효과는 후속 평가에서 확인할 목표 방향이며, 본 연구에서 검증한 정책 효과나 예측 수치가 아닙니다. 이직 고려 의향과 실제 이직률은 서로 다른 지표입니다.')
    section_title('최종 결론 및 실무 제언')
    panel('우선 진단에서 후속 평가까지',
          '기본 선별과 공변량 조정 결과를 종합하여 현장 진단의 우선순위를 정합니다. 신뢰·만족 관련 지표를 모니터링하고 소통의 관련성은 현장 맥락과 함께 확인합니다. 현장 진단에서 확인된 문제에 한해 비용·실행 가능성·직원 보호 기준을 검토한 뒤 시범사업을 평가합니다.',
          'teal','정책 ROI는 추정하지 않았으며, 개인별 위험 점수화나 자동 채용·평가·징계에 사용해서는 안 됩니다.')

# Unified source and navigation footer.
st.divider()
current_index=ALL_NAV_KEYS.index(page)
if current_index<len(ALL_NAV_KEYS)-1:
    next_page=ALL_NAV_KEYS[current_index+1]
    source_col,nav_col=st.columns([2.7,1.3],gap='medium')
    with source_col:
        source_line(plain=True)
        if page=='overview':
            overview_reference_footer()
    with nav_col:
        st.button('→',key='next_'+page,help=NAV_LABELS[next_page]+' 화면으로 이동',
                  use_container_width=False,on_click=move_to,args=(next_page,))
else:
    source_line(plain=True)

# Run after the destination page has rendered, not on filter/widget reruns.
scroll_to_page_top()
