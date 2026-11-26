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


def theory_container(key, card=False):
    """Use native containers; keep a bordered fallback for older Streamlit."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container(border=card)


def research_section_header(number,title,note=''):
    """Use the same numbered section hierarchy as the HR decision page."""
    st.markdown(
        f'<div class="decision-section-head"><span class="decision-section-number">{escape(number)}</span>'
        f'<div><h2>{escape(title)}</h2>'
        +(f'<p>{escape(note)}</p>' if note else '')
        +'</div></div>',unsafe_allow_html=True,
    )


def theory_cards(group, cards, columns):
    """Render the original Markdown text in responsive, scoped cards."""
    with theory_container('theory_grid_'+group):
        for start in range(0,len(cards),columns):
            cols=st.columns(columns,gap='medium')
            for index,(kicker,heading,body,tone) in enumerate(cards[start:start+columns]):
                with cols[index]:
                    with theory_container(f'theory_card_{group}_{start+index}',card=True):
                        st.markdown(
                            f'<div class="theory-card-kicker {tone}">{escape(kicker)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(heading)
                        st.markdown(body)


def theory_path_diagram():
    """Hypothesized directions only; no fitted coefficients or causal claims."""
    svg = '''
<svg xmlns="http://www.w3.org/2000/svg" width="720" height="240" viewBox="0 0 960 320" role="img" aria-label="H1–H5의 예상 방향: 조직신뢰와 의사소통에서 직무만족, 그리고 이직 고려 의향으로 연결되는 연구가설">
  <defs>
    <marker id="plusArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8 Z" fill="#2563EB"/></marker>
    <marker id="minusArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8 Z" fill="#0D9488"/></marker>
  </defs>
  <rect x="1" y="1" width="958" height="318" rx="14" fill="#FFFFFF" stroke="#E2E8F0"/>
  <g font-family="Pretendard, Noto Sans KR, Malgun Gothic, sans-serif" text-anchor="middle" font-weight="700">
    <g fill="none" stroke-width="2.2" stroke-linejoin="round">
      <path d="M248 82 L366 143" stroke="#2563EB" marker-end="url(#plusArrow)"/>
      <path d="M248 238 L366 177" stroke="#2563EB" marker-end="url(#plusArrow)"/>
      <path d="M142 48 L142 25 L822 25 L822 126" stroke="#0D9488" marker-end="url(#minusArrow)"/>
      <path d="M142 272 L142 295 L822 295 L822 194" stroke="#0D9488" marker-end="url(#minusArrow)"/>
      <path d="M580 160 L720 160" stroke="#0D9488" marker-end="url(#minusArrow)"/>
    </g>
    <g font-size="16">
      <text x="306" y="91" fill="#2563EB">H1 (+)</text>
      <text x="306" y="238" fill="#2563EB">H2 (+)</text>
      <text x="480" y="19" fill="#0D9488">H3 (−)</text>
      <text x="480" y="312" fill="#0D9488">H4 (−)</text>
      <text x="650" y="148" fill="#0D9488">H5 (−)</text>
    </g>
    <rect x="36" y="48" width="212" height="68" rx="11" fill="#EFF6FF" stroke="#BFDBFE"/>
    <rect x="36" y="204" width="212" height="68" rx="11" fill="#EFF6FF" stroke="#BFDBFE"/>
    <rect x="366" y="126" width="214" height="68" rx="11" fill="#F5F3FF" stroke="#C4B5FD"/>
    <rect x="720" y="126" width="204" height="68" rx="11" fill="#F0FDFA" stroke="#99F6E4"/>
    <g font-size="21" dominant-baseline="middle">
      <text x="142" y="82" fill="#1D4ED8">조직신뢰</text>
      <text x="142" y="238" fill="#1D4ED8">의사소통</text>
      <text x="473" y="160" fill="#6D28D9">직무만족</text>
      <text x="822" y="160" fill="#0F766E">이직 고려 의향</text>
    </g>
  </g>
</svg>'''
    with theory_container('theory_path_visual'):
        st.image(svg, width=720)
        st.markdown(
            '<div class="theory-indirect"><strong>H6 · 탐색적 간접경로 (−)</strong>'
            '<span>조직신뢰 → 직무만족 → 이직 고려 의향</span></div>',
            unsafe_allow_html=True,
        )
        st.caption('가설의 예상 방향을 나타낸 모형입니다. 실제 추정값이나 인과효과를 의미하지 않습니다.')


def research_text_sections():
    """Separate research design from result interpretation without editing the source."""
    lines=(BASE/'THEORY_AND_HYPOTHESES.md').read_text(encoding='utf-8').splitlines()
    if lines and lines[0].startswith('# '):
        lines=lines[1:]
    text='\n'.join(lines).lstrip()
    design_text,h3,outcomes=text.partition('### 3. 이론과 다르게 나타난 결과를 어떻게 해석하는가?')
    intro,h1,remaining=design_text.partition('### 1. 이론과 선행연구에서 가설을 도출하는 과정')
    foundations,h2,design=remaining.partition('### 2. 이론에서 연구모형으로: 무엇을 측정·통제했는가?')
    measurement,method_heading,methods=design.partition('**모형과 질문의 대응**:')
    discussion,h4,future=outcomes.partition('### 4. 대안 설명을 줄이기 위한 후속 연구 제안')
    future_steps,practice_heading,practice=future.strip().partition('\n\n**실무 연결:**')
    return dict(design_text=design_text.strip(),ready=bool(h1 and h2 and method_heading),
                intro=intro.strip(),foundations=foundations.strip(),measurement=measurement.strip(),
                methods=(method_heading+methods).strip(),discussion=discussion.strip(),
                future=future.strip(),step_lines=future_steps.strip().splitlines(),
                practice=practice.strip(),has_practice=bool(practice_heading))


def render_reference_information(reference_text):
    """Present the existing reference document by section, without rewriting it."""
    expected=['자료 출처','참고문헌','AI 활용 및 검증','연구윤리 및 활용 범위']
    sections=[]
    for chunk in reference_text.split('\n### '):
        if chunk.startswith('### '):
            chunk=chunk[4:]
        title,separator,body=chunk.partition('\n')
        sections.append((title.strip(),body.strip()))
    if [title for title,_ in sections]!=expected:
        st.markdown(reference_text)
        return
    with theory_container('research_reference_content'):
        tabs=st.tabs(expected)
        for index,((title,body),tab) in enumerate(zip(sections,tabs),1):
            with tab:
                with theory_container(f'research_reference_section_{index}',card=True):
                    research_section_header(f'{index:02d}',title)
                    st.markdown(body)


def research_reference_link():
    reference_text=(BASE/'REFERENCES_AND_AI.md').read_text(encoding='utf-8')
    reference_label='연구 정보'
    dialog=getattr(st,'dialog',None) or getattr(st,'experimental_dialog',None)
    if dialog is not None:
        @dialog('연구 참고 정보',width='large')
        def show_reference_information():
            render_reference_information(reference_text)
        # The container key scopes styling to this research reference control only.
        try:
            link_container=st.container(key='research_reference_link')
        except TypeError:
            link_container=st.container()
        with link_container:
            button_options=dict(key='research_reference_info',use_container_width=False,
                                help='자료 출처 · 참고문헌 · AI 활용 · 연구윤리')
            try:
                opened=st.button(reference_label,icon=':material/menu_book:',**button_options)
            except TypeError:
                opened=st.button(reference_label,**button_options)
            if opened:
                show_reference_information()
    else:
        with st.expander('자료 출처 · 참고문헌 · AI 활용 · 연구윤리',expanded=False):
            render_reference_information(reference_text)


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
    # Use spacing and separators rather than visible stage headings.
    # Keep the existing native navigation controls, groups and stable keys.
    for group_index,(_,items_in_group) in enumerate(NAV_GROUPS):
        with theory_container(f'sidebar_nav_group_{group_index}'):
            for item_key,item_label in items_in_group:
                is_active=(st.session_state['page']==item_key)
                st.button(item_label,key='nav_'+item_key,type='primary' if is_active else 'secondary',
                          use_container_width=True,on_click=move_to,args=(item_key,))
    st.markdown(
      '<div class="sidebar-footer"><div class="sidebar-meta">'
      '<div class="sidebar-info-label">분석 표본</div>'
      '<strong>10,731명 · 497개 기업</strong><br>'
      '개별 응답 · 기업 식별정보 없음</div>'
      '<div class="sidebar-meta sidebar-source">'
      '<div class="sidebar-info-label">DATA SOURCE</div>'
      '<strong>한국직업능력연구원(KRIVET)</strong><br>'
      'HCCPⅡ 제4차(2023) 근로자 조사'
      '<div class="sidebar-source-note">통계값은 연구자가 산출한 추정치</div>'
      '<a href="https://www.krivet.re.kr/kor/sub.do?menuSn=18" '
      'target="_blank" rel="noopener noreferrer">공식 자료 보기 ↗</a>'
      '</div></div>',unsafe_allow_html=True,
    )

page=st.session_state['page']
try:
    header_container=st.container(key='app_header')
except TypeError:
    header_container=st.container()
with header_container:
    try:
        header_main,header_info=st.columns([5.5,1],gap='small',vertical_alignment='center')
    except TypeError:
        header_main,header_info=st.columns([5.5,1],gap='small')
    with header_main:
        st.markdown(
          '<div class="app-topbar"><span class="app-topbar-mark">HR</span>'
          '<strong>Decision Analytics</strong><span class="topbar-divider"></span>'
          f'<span class="topbar-page">{escape(NAV_LABELS[page])}</span><span class="topbar-version">HCCPⅡ · 2023</span></div>',
          unsafe_allow_html=True,
        )
    with header_info:
        research_reference_link()

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

    with theory_container('research_section_overview_01',card=True):
        research_section_header('01',"연구 표본 및 범위","공통 분석표본과 공개 자료의 범위를 확인합니다.")
        a,b,c,d=st.columns(4,gap='medium')
        a.metric('분석 근로자',f'{P["analysis_rows"]:,}명')
        b.metric('대상 기업',f'{P["analysis_firms"]:,}개')
        c.metric('주 검증 가설','H1–H5')
        d.metric('조사 기준 연도','2023')
        st.caption('전체 응답자 10,759명 중 연령 결측 28명 제외 · 공개 범위: 집계 통계 및 사전 추정 결과(원자료 제외)')

    with theory_container('research_section_overview_02',card=True):
        research_section_header('02',"핵심 연구 결과","표본 10,731명에 대한 조건부 관련성 결과입니다. 정책 효과나 인과관계를 추정하지 않았습니다.")
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

    with theory_container('research_section_overview_03',card=True):
        research_section_header('03',"핵심 지표 및 연구 진행 흐름","관찰 평균에서 HR 의사결정으로 이어지는 연구 구조입니다.")
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
                '<div class="step-list overview-flow">'
                '<div><span>01</span><strong>실무질문과 이론</strong><small>어떤 조직요인이 직무만족·이직 고려와 관련되는가?</small></div>'
                '<div><span>02</span><strong>5단계 분석</strong><small>신뢰도 → 집단 평균 → OLS → 순서형 로짓 → SEM</small></div>'
                '<div><span>03</span><strong>HR 의사결정</strong><small>진단 후보 선별 → 현장 검증 → 제한적 시범사업 평가</small></div>'
                '</div>',unsafe_allow_html=True,
            )

elif page=='theory':
    page_intro('RESEARCH DESIGN  /  THEORY','이론적 배경 및 연구가설',
               '실무질문에서 이론과 연구가설을 도출하고, 측정변수·통제변수 및 분석 절차를 정리합니다.')
    research=research_text_sections()
    if research['ready']:
        with theory_container('theory_question',card=True):
            st.markdown(research['intro'])
        with theory_container('research_section_theory_01',card=True):
            research_section_header('01',"이론적 배경 및 선행연구","이론에 근거한 가설의 예상 방향입니다. 실제 검증 결과와 구분합니다.")
            foundations=research['foundations']
            foundation_cards=[
                ('사회교환이론','Blau (1964)',
                 '신뢰와 상호성에 기반한 관계에서 긍정적인 직무태도를 예상합니다.',
                 [('H1','조직신뢰 → 직무만족','+'),
                  ('H3','조직신뢰 → 이직 고려 의향','−')]),
                ('직무요구-자원(JD-R) 모형','Bakker & Demerouti (2007)',
                 '의사소통을 정보·지원 자원의 대리지표로 해석합니다.',
                 [('H2','의사소통 → 직무만족','+'),
                  ('H4','의사소통 → 이직 고려 의향','−')]),
                ('직무태도와 이직과정','Tett & Meyer (1993)',
                 '직무만족과 이직 고려 의향의 관계 및 탐색적 간접경로를 검토합니다.',
                 [('H5','직무만족 → 이직 고려 의향','−'),
                  ('H6','조직신뢰 → 직무만족 → 이직 고려 의향','간접 −')]),
            ]
            card_html=[]
            for title,author,summary,paths in foundation_cards:
                relations=[]
                for hid,path,direction in paths:
                    tone='purple' if hid=='H6' else 'blue' if direction=='+' else 'teal'
                    relations.append(
                        f'<div class="foundation-relation"><span class="foundation-hypothesis">{escape(hid)}</span>'
                        f'<span class="foundation-path">{escape(path)}</span>'
                        f'<span class="foundation-direction {tone}">{escape(direction)}</span></div>'
                    )
                card_html.append(
                    f'<article class="foundation-card"><div class="foundation-title">{escape(title)}</div>'
                    f'<div class="foundation-author">{escape(author)}</div>'
                    f'<div class="foundation-summary">{escape(summary)}</div>'
                    '<div class="foundation-relations">'+''.join(relations)+'</div></article>'
                )
            st.markdown('<div class="foundation-grid">'+''.join(card_html)+'</div>',unsafe_allow_html=True)
            with st.expander('이론적 근거 및 선행연구 상세 보기',expanded=False):
                st.markdown(foundations.strip())

        with theory_container('research_section_theory_02',card=True):
            research_section_header('02',"연구가설 및 연구모형","H1–H5의 예상 방향과 H6의 탐색적 간접경로를 나타냅니다.")
            theory_path_diagram()

        with theory_container('research_section_theory_03',card=True):
            research_section_header('03',"측정변수 및 통제변수","")
            st.markdown(research['measurement'])

        with theory_container('research_section_theory_04',card=True):
            research_section_header('04',"분석 방법 및 검증 절차","측정도구 검토에서 가설 검증까지 순차적으로 진행합니다.")
            st.markdown(
                '<div class="theory-method-flow">'
                '<div><span>01</span><strong>측정 신뢰도</strong><small>문항 및 측정모형 점검</small></div>'
                '<div><span>02</span><strong>기술통계 · 집단 비교</strong><small>관찰 평균 및 상관 확인</small></div>'
                '<div><span>03</span><strong>다중회귀 OLS</strong><small>H1–H5 주 검증</small></div>'
                '<div><span>04</span><strong>순서형 로짓</strong><small>대안 모형 및 가정 점검</small></div>'
                '<div><span>05</span><strong>구조방정식 SEM</strong><small>구조경로 및 H6 탐색</small></div>'
                '</div>',unsafe_allow_html=True,
            )
            st.markdown(research['methods'])

    else:
        st.markdown(research['design_text'])

elif page=='reliability':
    page_intro('DATA ANALYSIS  /  01','측정 신뢰도 및 타당성','조직신뢰·의사소통 문항의 내적 일관성을 살펴보고, 두 구성개념을 구분할 수 있는지 점검합니다.')
    with theory_container('research_section_reliability_01',card=True):
        research_section_header('01',"측정 신뢰도 및 타당성 진단","내적 일관성과 판별타당성의 주요 결과를 구분합니다.")
        c1,c2,c3=st.columns(3)
        c1.metric('조직신뢰 · Cronbach α',f'{P["alpha_trust"]:.3f}')
        c2.metric('의사소통 · Cronbach α',f'{P["alpha_comm"]:.3f}')
        c3.metric('2요인 CFA · 요인상관',f'{SA["latent_correlation"]:.3f}')
        panel('측정 타당성 · 판별타당성 미확보',
              f'두 요인 간 상관이 {SA["latent_correlation"]:.3f}이며, 2요인 CFA의 CFI {SA["two_factor"]["cfi"]:.3f}, RMSEA {SA["two_factor"]["rmsea"]:.3f}는 측정모형의 한계를 보여줍니다.',
              'amber','내적 일관성이 양호하더라도 조직신뢰와 의사소통을 통계적으로 명확하게 구분하기는 어렵습니다.')

    with theory_container('research_section_reliability_02',card=True):
        research_section_header('02',"문항별 5점 평균","")
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

    with theory_container('research_section_reliability_03',card=True):
        research_section_header('03',"CFA 측정모형 적합도","5점 문항을 연속형으로 근사한 ML 적합 결과이며 기업 군집·순서형 분포는 보정하지 않았습니다.")
        cfa=pd.DataFrame([{'모형':name,'CFI':d['cfi'],'TLI':d['tli'],'RMSEA':d['rmsea'],'SRMR':d['srmr']}
           for name,d in [('1요인',SA['one_factor']),('2요인',SA['two_factor'])]])
        st.dataframe(cfa.round(3),use_container_width=True,hide_index=True)
        st.markdown('**측정 결과 해석** · 1요인·2요인 모형 모두 RMSEA가 높고, 2요인 모형의 적합도 개선이 확인되지 않았습니다.')
        st.caption('후속 검토 | 조직신뢰와 의사소통을 구분해 해석하기 전에 문항 구조와 순서형·기업 군집 보정을 추가로 검토할 필요가 있습니다.')

elif page=='descriptive':
    page_intro('DATA ANALYSIS  /  02','기술통계 및 집단 비교','사전 집계된 통계만 조회할 수 있으며, 개인 단위 필터·역추적 가능한 조합 조회는 제공하지 않습니다.')
    with theory_container('research_section_descriptive_01',card=True):
        research_section_header('01',"핵심 변수의 전체 평균","공통표본의 5점 척도 관찰 평균입니다.")
        cols=st.columns(4)
        for (key,name),col in zip(metrics.items(),cols):
            col.metric(name,f'{P["overall_means"][key]:.2f} / 5')

    with theory_container('research_section_descriptive_02',card=True):
        research_section_header('02',"집단별 평균 비교","집단 분류와 관심 지표를 선택하면 공개 기준을 통과한 사전 집계값만 표시됩니다.")
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

    with theory_container('research_section_descriptive_03',card=True):
        research_section_header('03',"핵심 변수의 관찰 상관","전체 공통표본의 단순 상관과 기업 단위 부트스트랩 95% 신뢰구간입니다.")
        plot(viz.observed_associations(OBS),key='observed_correlations')
        st.markdown('**집단 비교의 활용** · 집단별 평균과 변수 간 상관은 현장 진단의 탐색적 근거입니다. 평균 순위만으로 판단하지 않고 HR 의사결정 페이지의 집단 규모·신뢰구간·다중비교 기준을 함께 확인합니다.')
        st.caption('집단 평균 차이를 개인의 특성이나 정책 시행 효과로 해석하지 않습니다.')

elif page=='ols':
    page_intro('DATA ANALYSIS  /  03','다중회귀분석 · H1–H5','개인·직무·기업 특성을 통제하고 기업 단위 군집강건 표준오차를 적용한 다중회귀분석 결과입니다.')
    with theory_container('research_section_ols_01',card=True):
        research_section_header('01',"가설별 회귀계수 및 신뢰구간","B는 다른 변수와 공변량을 통제한 조건부 관련성입니다.")
        c1,c2,c3=st.columns(3)
        c1.metric('표본',f'{P["analysis_rows"]:,}명')
        c2.metric('기업 단위 군집',f'{P["analysis_firms"]:,}개')
        c3.metric('핵심 연구가설','H1–H5')
        st.dataframe(ols_table(),hide_index=True,use_container_width=True)
        st.markdown('**분석할 종속변수**')
        dv=st.radio('종속변수 선택',['직무만족','이직 고려 의향'],horizontal=True,label_visibility='collapsed')
        model='직무만족 전체 통제' if dv=='직무만족' else '전체 통제'
        plot(viz.regression_forest(CO.loc[CO['model']==model],{**metrics,**items}),key='ols_forest_'+model)
        st.caption(f'{model} 모형의 R² = {S["models"][model]["r2"]:.3f}. OLS 신뢰구간은 기업 단위 군집강건 표준오차에 기반합니다.')

    with theory_container('research_section_ols_02',card=True):
        research_section_header('02',"의사소통 계수 변화","모형에 추가되는 설명변수·통제변수에 따라 부호가 달라질 수 있습니다.")
        plot(viz.regression_change(CO),key='ols_coefficient_change')
        panel('회귀분석 결과 해석',
              'OLS에서 H1·H2·H3·H5는 가설의 예상 방향과 일치했으며, H4는 예상과 반대로 양(+)의 관련성을 보였습니다.',
              'teal','신뢰·만족 관련 지표를 우선 모니터링하고, 소통의 방향성은 조직 맥락과 함께 검토합니다.')

    with theory_container('research_section_ols_03',card=True):
        research_section_header('03',"분석 가정 및 해석상의 한계","확인된 사항과 추가 검증이 필요한 항목을 구분하여 분석 결과의 해석 범위를 제시합니다.")
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
    with theory_container('research_section_ordinal_01',card=True):
        research_section_header('01',"이직 고려 의향 · 순서형 로짓","OR>1은 더 높은 이직 고려 응답범주의 누적 오즈와 양(+)의 관련성입니다.")
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

    with theory_container('research_section_ordinal_02',card=True):
        research_section_header('02',"직무만족 · 순서형 로짓","직무만족을 결과변수로 하는 보완적 결과로, 비례오즈 가정에 대한 별도 검정은 수행되지 않았습니다.")
        st.dataframe(pd.DataFrame([{'요인':e['label'],'OLS B':fmt(e['ols_coef']),
           '순서형 로짓 B':fmt(e['ordinal_coef']),'OR':round(e['odds_ratio'],3),
           'OR 95% CI':f'[{e["or_ci_low"]:.3f}, {e["or_ci_high"]:.3f}]'} for e in OS['findings']]),hide_index=True,use_container_width=True)

    with theory_container('research_section_ordinal_03',card=True):
        research_section_header('03',"이직 고려 의향 모형의 비례오즈 가정 검토","조직신뢰·의사소통·직무만족의 응답 경계별 계수 동일성을 Wald 검정으로 점검했습니다. 전체 변수에 대한 검정은 아닙니다.")
        w=OR['threshold_analysis']['focus_slope_equality_wald']
        diag=pd.DataFrame([{'변수':row['label'],'Wald χ²':round(row['wald_chi2'],3),'df':row['df'],
          'p':'<0.001' if row['p']<.001 else f'{row["p"]:.3f}',
          '해석':'경계별 동일성 위배 가능성' if row['p']<.05 else '위배 근거가 확인되지 않음'} for row in w['per_factor']])
        st.dataframe(diag,hide_index=True,use_container_width=True)
        st.markdown('**강건성 점검 결과** · 순서형 로짓에서도 H1–H5의 OLS 계수 방향이 일관되게 유지되었습니다. 다만 이직 고려 의향 모형의 비례오즈 가정에서 일부 위배 신호가 확인되었으므로, 결과의 방향적 일관성은 확인되었으나 모형의 강건성이 완전히 확보되었다고 단정할 수 없습니다.')
        st.caption('해석 유의사항 | 이직 고려 의향 모형에서 직무만족의 경계별 계수 동일성 위배 신호가 있으며, 의사소통 오즈비의 95% 신뢰구간 하한은 1에 가깝습니다. 직무만족을 결과변수로 하는 모형의 비례오즈 가정은 별도로 검증하지 않았습니다.')

elif page=='sem':
    page_intro('DATA ANALYSIS  /  05','구조방정식 · SEM','H1–H5의 보완적 구조경로와 탐색적 간접경로 H6를 실제 저장된 추정값으로 보여줍니다.')
    with theory_container('research_section_sem_01',card=True):
        research_section_header('01',"SEM 모형 적합도 및 해석 범위","탐색적 혼합형 SEM의 적합도를 여러 지표로 종합하여 판단합니다.")
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

    with theory_container('research_section_sem_02',card=True):
        research_section_header('02',"SEM 구조모형 및 경로도","조직신뢰는 3문항 잠재요인, 의사소통은 4문항 평균 관측 합성지표로 처리했습니다. 탐색적 구조경로이며 인과관계를 뜻하지 않습니다.")
        plot(viz.sem_fitted_paths(SEM),key='sem_structure')

    with theory_container('research_section_sem_03',card=True):
        research_section_header('03',"가설별 경로계수 및 신뢰구간","기업 단위 부트스트랩 95% 신뢰구간을 제시합니다.")
        st.dataframe(pd.DataFrame([{'가설':h['id'],'경로':h['path'],'예상':h['direction'],
           'SEM B':fmt(SEM_EFFECTS[h['id']]['estimate']),
           '95% CI':f'[{fmt(SEM_EFFECTS[h["id"]]["ci_low"])}, {fmt(SEM_EFFECTS[h["id"]]["ci_high"])}]',
           '판정':SEM_EFFECTS[h['id']]['classification']} for h in SEM_HYPOTHESES]),
           hide_index=True,use_container_width=True)

    with theory_container('research_section_sem_04',card=True):
        research_section_header('04',"직접·간접·전체 경로의 구분","H6는 모형에서 추정한 통계적 간접경로이며 시간적 매개효과를 입증하지 않습니다.")
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
    research=research_text_sections()
    with theory_container('decision_group_01',card=True):
        st.markdown(
            '<div class="decision-section-head"><span class="decision-section-number">01</span>'
            '<div><h2>연구 결과 종합 해석</h2><p>분석을 통해 무엇을 확인했는가?</p></div></div>',
            unsafe_allow_html=True,
        )
        findings=[
            ('H1 · H3 · H5','이론적 기대와 방향 일치','teal',
             '조직신뢰–직무만족 (+), 조직신뢰·직무만족–이직 고려 의향 (−)',
             '이론적 기대와 양립하는 통계적 관련성입니다. 조직신뢰 개선 정책의 효과가 입증된 것은 아닙니다.'),
            ('H2','모형 간 부호 불일치','amber',
             '의사소통–직무만족: OLS (+), SEM (−)',
             '신뢰와 소통의 높은 상관, 측정구조 중첩, 통제변수와 측정 단위의 차이가 조건부 계수 민감성의 가능한 설명입니다. JD-R의 자원 경로가 확인되거나 반증되었다고 단정할 수 없습니다.'),
            ('H4','예상과 반대','amber',
             '의사소통–이직 고려 의향: OLS (+)',
             '문제가 많은 부서의 소통·보고 증가, 이직을 고려하는 근로자의 소통 평가 차이 등 역인과와 업무요구·관리자 특성의 누락, 구성개념 중첩이 경쟁 설명입니다. 의사소통이 이직을 증가시킨다는 결론은 내릴 수 없습니다.'),
            ('H6','탐색적 간접경로','purple',
             '조직신뢰 → 직무만족 → 이직 고려 의향 (−)',
             'SEM에서 음의 간접경로가 나타났으나 단면 설문, 측정 한계 및 높은 구성개념 상관으로 인과적 매개효과를 확증할 수 없습니다.'),
        ]
        finding_rows=[]
        for hypothesis,status,tone,observed,meaning in findings:
            finding_rows.append(
                '<article class="decision-finding-row">'
                f'<div class="decision-finding-label"><strong>{escape(hypothesis)}</strong>'
                f'<span class="decision-finding-status {tone}">{escape(status)}</span></div>'
                f'<div class="decision-finding-result"><small>확인된 관계</small><strong>{escape(observed)}</strong></div>'
                f'<div class="decision-finding-meaning"><small>해석 및 대안 설명</small><p>{escape(meaning)}</p></div>'
                '</article>'
            )
        st.markdown(
            '<div class="decision-finding-matrix">'
            '<div class="decision-finding-head"><span>가설 및 판정</span><span>확인된 관계</span><span>해석 및 대안 설명</span></div>'
            +''.join(finding_rows)+'</div>',
            unsafe_allow_html=True,
        )
        st.caption('가설별 추정값과 신뢰구간은 03 다중회귀 OLS, 04 순서형 로짓, 05 구조방정식 SEM에서 확인할 수 있습니다. 대안 설명은 가능성이며 실제 원인이 검증된 것은 아닙니다.')

    with theory_container('decision_group_02',card=True):
        st.markdown(
            '<div class="decision-section-head"><span class="decision-section-number">02</span>'
            '<div><h2>우선 진단 대상 선정</h2><p>어떤 집단을 우선 살펴야 하는가?</p></div></div>',
            unsafe_allow_html=True,
        )
        selected=PR['selected_at_default']
        followups=PR.get('adjusted_selected_follow_up',[])
        customer_support=next((row for row in selected
                               if row['dimension']=='직무' and row['group_name']=='대고객지원'
                               and row['metric_name']=='이직 고려 의향'),None)
        adjusted=next((row for row in followups
                       if row['field']=='W23Q04' and row['code']==5 and row['metric']=='turnover'),None)
        st.markdown(
            '<div class="decision-screening">'
            f'<div><small>사전 비교</small><strong>{PR["contrasts"]}건</strong></div>'
            f'<div><small>기본 기준 충족</small><strong>{len(selected)}건</strong></div>'
            '<div class="decision-screening-rule"><small>기본 선별 기준</small>'
            '<p>불리한 평균 차이 ≥ 0.3점 · 응답자 ≥ 50명 · 기업 ≥ 10개<br>95% CI 하한 &gt; 0 · BH 보정 q &lt; 0.05</p></div>'
            '</div>',unsafe_allow_html=True,
        )
        if len(selected)==1 and customer_support is not None and adjusted is not None:
            st.markdown(
                '<div class="decision-candidate-summary">'
                '<div class="candidate-identity"><small>우선 현장 진단 후보</small>'
                f'<strong>{escape(customer_support["group_name"])} 직무</strong>'
                f'<p>{escape(customer_support["metric_name"])} · 응답자 {customer_support["n"]:,}명 · {customer_support["firms_in_group"]:,}개 기업</p></div>'
                '<div class="candidate-comparison">'
                '<div><small>집단 대 나머지 평균 차이</small>'
                f'<strong>{fmt(customer_support["adverse_difference"])}점</strong>'
                f'<p>95% CI [{fmt(customer_support["adverse_ci_low"])}, {fmt(customer_support["adverse_ci_high"])}]</p>'
                f'<p>BH-FDR q = {customer_support["q_bh_152"]:.4f}</p>'
                '<small>기본 선별 기준에 따른 후보 도출</small></div>'
                '<div><small>공변량 조정 후 회귀계수</small>'
                f'<strong>{fmt(adjusted["adjusted_coef"])}점</strong>'
                f'<p>95% CI [{fmt(adjusted["adjusted_ci"][0])}, {fmt(adjusted["adjusted_ci"][1])}]</p>'
                f'<p>p = {adjusted["adjusted_p"]:.4f}</p>'
                '<small>집단 구성·근무 여건을 고려한 탐색적 검토</small></div>'
                '</div><div class="candidate-judgment"><strong>선정 근거</strong>'
                '<p>공변량 조정 후 차이의 크기는 작아졌지만 양(+)의 방향이 유지되고 신뢰구간에 0을 포함하지 않습니다. 현장 확인 후보이며 투자·개입 확정 대상은 아닙니다.</p></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(pd.DataFrame([{'집단':row['group_name'],'분류':row['dimension'],'지표':row['metric_name'],
               '응답자':row['n'],'기업':row['firms_in_group'],
               '불리한 평균 차이':round(row['adverse_difference'],3),
               '95% CI':f'[{row["adverse_ci_low"]:+.3f}, {row["adverse_ci_high"]:+.3f}]',
               'BH 보정 q':round(row['q_bh_152'],4)} for row in selected]),hide_index=True,use_container_width=True)
            if customer_support is not None and adjusted is not None:
                section_title('우선 진단 후보의 추가 분석')
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
        if customer_support is not None and adjusted is not None:
            with st.expander('선정 기준 및 공변량 조정 해석 범위',expanded=False):
                st.markdown('공변량 조정은 후보 선별 후 탐색적 분석입니다. p값을 다중검정·선별 보정된 확증적 근거로 해석하지 않습니다. 기본 0.3점 기준은 최초 평균 차이에만 적용합니다. 조정 전후 차이는 특정 원인이나 직무의 인과효과를 증명하지 않습니다.')
                st.caption(adjusted.get('control_note',''))
        section_title('선별 기준 민감도 점검','기본 후보를 변경하는 기능이 아니라 사전 계산된 조건별 통과 건수 조회입니다.')
        with theory_container('decision_sensitivity'):
            c1,c2,c3=st.columns([1,1,0.85],gap='medium')
            cutoff=c1.select_slider('불리한 평균 차이 기준(점)',options=[0.2,0.3,0.4],value=0.3)
            minimum=c2.select_slider('집단 내 최소 응답자 수',options=[30,50,100],value=50)
            sensitivity=PR['sensitivity_counts'][f'g{cutoff:.1f}_n{minimum}']
            c3.metric('선별되는 집단–지표 수',f'{int(sensitivity)}건')
        st.caption('다른 조건에서 집단 이름을 추정하지 않으며, 위 기본 기준의 후보 및 추가 분석 결과는 변경되지 않습니다. 공개 평균표 기준(100명·15개 기업)과 진단 후보 선별 기준은 별개입니다.')

    with theory_container('decision_group_03',card=True):
        st.markdown(
            '<div class="decision-section-head"><span class="decision-section-number">03</span>'
            '<div><h2>HR 의사결정 및 실행 계획</h2><p>실무에서 어떻게 활용할 것인가?</p></div></div>',
            unsafe_allow_html=True,
        )
        section_title('HR 실행 3단계')
        st.markdown(
            '<div class="decision-execution-flow">'
            '<article><small>01 · HR 분석·기획</small><strong>진단 후보 확인</strong><p>집단 규모·신뢰구간·다중비교·차별 위험 검토</p></article>'
            '<article><small>02 · 현업 리더</small><strong>현장 원인 확인</strong><p>익명 의견 수렴 및 업무 프로세스 점검</p></article>'
            '<article><small>03 · 경영진·HR 의사결정자</small><strong>시범사업 평가</strong><p>목표지표·기간·예산·사전/사후 평가</p></article>'
            '</div>',unsafe_allow_html=True,
        )
        if customer_support is not None and adjusted is not None:
            st.markdown('**현장 확인 항목** · 업무량, 교대근무, 고객응대 부담 및 조직 지원을 익명 조사로 확인하고, 확인된 원인에 한해 개선 과제를 검토합니다.')
        section_title('담당자별 의사결정 지원','담당자와 관심 지표를 선택하면 해당 실행 과제와 분석 근거를 확인할 수 있습니다. 통계값은 새로 계산되지 않습니다.')
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
        st.markdown(
            '<div class="decision-role-result">'
            '<article><small>담당자 · 다음 행동</small>'
            f'<strong>{escape(role)}</strong><p>{escape(action)}</p>'
            f'<div class="role-result-note"><b>확인 사항</b> {escape(gate)}</div></article>'
            '<article><small>관심 지표 · 분석 근거</small>'
            f'<strong>{escape(title)}</strong><p>{escape(evidence)}</p>'
            f'<div class="role-result-note"><b>해석 범위</b> {escape(caveat)}</div></article>'
            '</div>',unsafe_allow_html=True,
        )
        section_title('후속 평가 지표 및 목표 방향','현장 진단을 거쳐 개선 과제가 확인된 경우에 적용할 시범사업 평가 계획입니다.')
        st.dataframe(pd.DataFrame([
            {'구분':'핵심','평가 지표':'이직 고려 의향','확인 방법':'동일 5점 문항으로 사전·사후 설문','목표 방향':'감소'},
            {'구분':'핵심','평가 지표':'직무만족','확인 방법':'동일 5점 문항으로 반복 측정','목표 방향':'향상'},
            {'구분':'보조','평가 지표':'조직신뢰','확인 방법':'동일 3개 문항 및 익명 의견 수렴','목표 방향':'향상'},
            {'구분':'보조','평가 지표':'의사소통','확인 방법':'동일 4개 문항 및 업무 흐름 점검','목표 방향':'확인된 소통 문제 개선'},
            {'구분':'장기','평가 지표':'실제 자발적 이직률','확인 방법':'동일 기간 자발적 퇴사자 수 ÷ 평균 재직자 수','목표 방향':'감소'},
        ]),hide_index=True,use_container_width=True)
        st.caption('평가 계획 | HR은 시범사업 시작 전 기준값·평가 기간·성공 및 중단 기준을 정하고, 현업 리더는 익명 설문과 업무 맥락을 확인합니다. 가능하면 비교집단과 동일한 시점에 반복 측정하고, 경영진은 결과·비용·직원 보호 여부를 종합하여 확대 또는 중단을 판단합니다.')
        st.caption('기대효과는 후속 평가에서 확인할 목표 방향이며, 본 연구에서 검증한 정책 효과나 예측 수치가 아닙니다. 이직 고려 의향과 실제 이직률은 서로 다른 지표입니다.')

    with theory_container('decision_group_04',card=True):
        st.markdown(
            '<div class="decision-section-head"><span class="decision-section-number">04</span>'
            '<div><h2>연구의 한계 및 후속 연구</h2><p>연구의 한계와 이를 보완하기 위한 검증 과제를 연결합니다.</p></div></div>',
            unsafe_allow_html=True,
        )
        limitation_pairs=[
            ('자료 및 인과관계',
             '2023년 단면 조사이며 조사 가중치를 적용하지 않았습니다. 시간적 선후, 미관측 교란 및 대표성에 주의해야 합니다.',
             '종단자료 및 대표성 검토',
             '반복 조사에서 조직신뢰·소통을 먼저 측정하고 이후 직무만족·이직 고려 의향, 가능하면 실제 퇴사와 연결합니다. 가중치 적용 가능성과 대표성도 점검합니다.'),
            ('측정 타당성',
             '조직신뢰와 의사소통의 구성개념 중첩이 확인되었습니다. 직무만족과 이직 고려 의향은 단일문항이며, 이직 고려 의향은 실제 퇴사율과 다릅니다.',
             '측정도구 및 누락변수 보완',
             '업무요구·관리자 지원·고충 보고·소통 만족을 분리해 측정합니다. 신뢰와 소통의 판별타당성(CFA)을 재점검하고 직무만족·이직 고려 의향의 다문항 측정 가능성을 검토합니다.'),
            ('분석 모형',
             'OLS의 전체 VIF·정식 잔차 진단은 미확인입니다. 순서형 로짓의 비례오즈 가정에서 일부 위배 신호가 있으며, SEM 적합도와 판별타당성에도 한계가 있습니다.',
             '모형 가정 및 적합도 추가 검증',
             '원자료로 전체 VIF·잔차·영향점을 점검합니다. 직무만족 결과 모형을 포함해 비례오즈 가정을 추가 검토하고, SEM의 대안 측정모형과 적합도를 비교합니다.'),
            ('실무 적용',
             '후보 선별 후 추가 분석은 탐색적입니다. 개인별 이직 위험이나 개선 정책의 인과효과·ROI는 추정하지 않았습니다.',
             '독립적 진단 및 정책 평가',
             '재조사로 진단 후보의 관련성을 확인하고 자발적 현장 조사를 실시합니다. 개입 효과는 별도의 시범사업에서 사전·사후 측정 및 가능한 비교집단을 통해 평가합니다.'),
        ]
        limit_rows=[]
        for idx,(category,limitation,task,proposal) in enumerate(limitation_pairs,1):
            limit_rows.append(
                '<article class="limit-pair-row">'
                f'<div class="limit-pair-category"><span>{idx:02d}</span><strong>{escape(category)}</strong></div>'
                f'<div class="limit-pair-current"><small>연구의 한계</small><p>{escape(limitation)}</p></div>'
                f'<div class="limit-pair-next"><small>후속 연구 방향</small><strong>{escape(task)}</strong><p>{escape(proposal)}</p></div>'
                '</article>'
            )
        st.markdown(
            '<div class="limit-pair-matrix">'
            '<div class="limit-pair-header"><span>검토 영역</span><span>연구의 한계</span><span>후속 연구 방향</span></div>'
            +''.join(limit_rows)+'</div>',
            unsafe_allow_html=True,
        )
        st.caption('한계는 기존 분석에서 확인된 범위입니다. 후속 연구는 제안 단계이며 추가 검증을 수행한 결과가 아닙니다. 세부 점검은 측정 신뢰도, 다중회귀 OLS, 순서형 로짓, 구조방정식 SEM에서 확인할 수 있습니다.')

    with theory_container('decision_group_05',card=True):
        st.markdown(
            '<div class="decision-section-head"><span class="decision-section-number">05</span>'
            '<div><h2>최종 결론 및 실무 제언</h2><p>연구질문에 대한 답변과 학술적·실무적 의의를 정리합니다.</p></div></div>',
            unsafe_allow_html=True,
        )
        if customer_support is not None and adjusted is not None:
            recommendation=(
                f'{customer_support["group_name"]} 직무를 우선 진단 후보로 검토합니다. '
                '조직신뢰와 직무만족을 지속적으로 살피되, 의사소통은 현장 맥락에 따라 해석하고 '
                '실제 원인이 확인된 문제에 한해 개선 여부를 판단합니다.'
            )
        else:
            recommendation=(
                '기본 선별 기준을 충족한 집단을 현장 진단 후보로 검토합니다. '
                '조직신뢰와 직무만족을 지속적으로 살피되, 의사소통은 현장 맥락에 따라 해석하고 '
                '실제 원인이 확인된 문제에 한해 개선 여부를 판단합니다.'
            )
        st.markdown(
            '<div class="decision-conclusion">'
            '<div class="conclusion-answer">'
            '<div class="conclusion-eyebrow">연구질문에 대한 최종 답변</div>'
            '<div class="conclusion-headline">조직신뢰와 직무만족은 구성원 유지 진단에서 주목할 지표입니다.</div>'
            '<p>조직신뢰는 직무만족과 양(+)의 관련성을, 조직신뢰와 직무만족은 이직 고려 의향과 음(−)의 관련성을 보였습니다. 의사소통은 모형과 통제 조건에 따라 해석에 주의가 필요합니다. 이는 관찰된 통계적 관련성이며 인과효과를 의미하지 않습니다.</p>'
            '</div>'
            '<div class="conclusion-insights">'
            '<article><div class="conclusion-eyebrow">연구의 학술적·실무적 의의</div>'
            '<strong>가설 검증에서 진단 근거로</strong>'
            '<p>주 분석과 보완 모형을 비교해 일관된 관련성과 모형에 민감한 관계를 구분하고, 탐색적 간접경로를 검토했습니다. 집단별 분석을 함께 활용하여 조직요인과 우선 진단 후보를 연결했다는 점에 의의가 있습니다.</p></article>'
            '<article><div class="conclusion-eyebrow">HR 의사결정에 대한 최종 제언</div>'
            '<strong>현장 근거를 바탕으로 개선 여부 판단</strong>'
            f'<p>{escape(recommendation)}</p></article>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        with theory_container('decision_conclusion_scope'):
            if research['has_practice']:
                st.markdown('**연구의 활용 범위:** '+research['practice'])
            st.caption('정책 ROI는 추정하지 않았으며, 개인별 위험 점수화나 자동 채용·평가·징계에 사용해서는 안 됩니다.')

# Run after the destination page has rendered, not on filter/widget reruns.
scroll_to_page_top()
