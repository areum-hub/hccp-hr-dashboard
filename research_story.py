"""Research plan alignment: OLS primary H1-H5, ordinal robustness, SEM supplementary."""
from html import escape

PROJECT_TITLE = '조직 내 신뢰와 의사소통이 구성원의 직무만족 및 이직의도와 갖는 관계 분석'
PROJECT_SCOPE = 'HCCPⅡ 4차(2023) 근로자 조사 · 공통 분석표본 10,731명 · 497개 기업'
MAIN_QUESTION = (
    '조직 내 신뢰와 의사소통 수준은 구성원의 직무만족 및 이직의도와 어떠한 관계가 있으며, '
    'HR 담당자는 구성원 유지와 조직건강 개선을 위해 어떤 조직요인과 구성원 집단을 우선적으로 살펴야 하는가?'
)
RESEARCH_QUESTIONS = [
    {'id': 'RQ1', 'question': '조직신뢰와 의사소통은 직무만족과 어떤 관계를 갖는가?',
     'evidence': 'H1·H2: 다중회귀(OLS 추정·군집강건) → 보조 SEM', 'screen': '03 → 05'},
    {'id': 'RQ2', 'question': '조직신뢰·의사소통·직무만족은 이직의도와 어떤 관계를 갖는가?',
     'evidence': 'H3~H5: 다중회귀(OLS 추정·군집강건) → 순서형 로짓 → 보조 SEM', 'screen': '03 → 04 → 05'},
    {'id': 'RQ3', 'question': '어떤 조직요인과 구성원 집단을 우선 진단할 것인가?',
     'evidence': '기술통계·집단별 비교 → 회귀 결과·한계 종합', 'screen': '02 → 05'},
    {'id': '추가', 'question': '직무만족을 통한 간접경로도 관찰되는가?',
     'evidence': '계획 이후 추가한 탐색적 SEM 간접경로(H6)', 'screen': '05'},
]


def research_map_html():
    cards=[]
    for q in RESEARCH_QUESTIONS:
        cards.append('<div class="rq-map-row">'
                     f'<span class="rq-map-id">{escape(q["id"])}</span>'
                     '<div class="rq-map-copy">'
                     f'<strong>{escape(q["question"])}</strong>'
                     f'<small>{escape(q["evidence"])}</small>'
                     '</div>'
                     f'<span class="rq-map-screen">{escape(q["screen"])}</span>'
                     '</div>')
    return '<div class="rq-map">'+''.join(cards)+'</div>'


def ols_hypothesis_table_html(coefs):
    """Use the original middle-term H1-H5 as the primary, two-outcome OLS tests."""
    hypotheses=[
        ('H1','직무만족','조직신뢰','직무만족 전체 통제','trust',+1),
        ('H2','직무만족','의사소통','직무만족 전체 통제','comm',+1),
        ('H3','이직의도','조직신뢰','전체 통제','trust',-1),
        ('H4','이직의도','의사소통','전체 통제','comm',-1),
        ('H5','이직의도','직무만족','전체 통제','satisfaction',-1),
    ]
    rows=[]
    for h,outcome,explanatory,model,key,sign in hypotheses:
        s=coefs[(coefs['model']==model)&(coefs['variable']==key)]
        if len(s)!=1:
            raise ValueError(f'OLS result missing or duplicated: {h}')
        a=s.iloc[0]
        value=float(a['coef']); lo=float(a['ci_low']); hi=float(a['ci_high']); p=float(a['p'])
        aligned=(value*sign)>0
        sig=(lo>0 or hi<0)
        verdict=('예상 방향 일치' if aligned else '예상과 반대')+(' · CI ≠ 0' if sig else ' · 불확실')
        color='match' if aligned and sig else 'reverse'
        pp='<0.001' if p<.001 else f'{p:.3f}'
        rows.append('<tr>'
                    f'<td><span class="id">{h}</span></td>'
                    f'<td class="path">{outcome} · {explanatory}</td>'
                    f'<td class="sign">{"+" if sign>0 else "−"}</td>'
                    f'<td class="number">{value:+.3f}</td>'
                    f'<td class="ci-text">[{lo:+.3f}, {hi:+.3f}]</td>'
                    f'<td>{pp}</td>'
                    f'<td><span class="verdict {color}">{verdict}</span></td>'
                    '</tr>')
    return ('<div class="sem-results ols-results"><table><thead><tr><th>가설</th><th>종속변수 · 설명변수</th>'
            '<th>예상</th><th>OLS B</th><th>95% 군집강건 CI</th><th>p</th><th>결과</th>'
            '</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>')


def conclusion_by_question_html(sem, priority, coefs=None, ordinal=None):
    """Synthesize OLS primary evidence, ordinal robustness, and SEM caveats."""
    effects={e['key']:e for e in sem['effects']}
    def effect(k):
        e=effects[k]
        return f'{e["estimate"]:+.3f} (95% CI [{e["ci_low"]:+.3f}, {e["ci_high"]:+.3f}])'
    if coefs is None:
        # Backward-compatible summary for legacy scripts. New dashboard supplies OLS.
        satisfaction='직무만족의 조직신뢰·의사소통 관련 경로는 본문의 OLS 결과와 함께 해석합니다.'
        turnover='이직의도 관련 경로는 OLS 결과와 순서형 로짓 결과를 함께 확인합니다.'
    else:
        def b(model,variable):
            s=coefs[(coefs.model==model)&(coefs.variable==variable)]
            return float(s.iloc[0].coef)
        satisfaction=(f'주 분석인 다중회귀모형(OLS 추정·기업 군집강건)에서 조직신뢰 B={b("직무만족 전체 통제","trust"):+.3f}, '
                      f'의사소통 B={b("직무만족 전체 통제","comm"):+.3f}로 두 변수 모두 직무만족과 정(+)의 조건부 관련성을 보였습니다. '
                      f'보조 SEM에서는 조직신뢰 H1 {effect("H1")}였으나 의사소통 H2 {effect("H2")}로 부호가 달라졌습니다.')
        turnover=(f'OLS에서 조직신뢰 B={b("전체 통제","trust"):+.3f}, '
                  f'직무만족 B={b("전체 통제","satisfaction"):+.3f}는 이직의도와 음(-)의 관련성, '
                  f'의사소통 B={b("전체 통제","comm"):+.3f}는 양(+)의 관련성을 보였습니다. '
                  '순서형 로짓에서도 해당 세 요인의 방향은 같지만, 의사소통 결과는 경계적이며 비례오즈 가정에 주의해야 합니다.')
    groups=priority.get('selected_at_default',[])
    if groups:
        first=groups[0]
        screening=(f'기본 선별 기준을 충족한 집단×지표 조합은 {len(groups)}건입니다. '
                   f'예: {escape(first["group_name"])}의 {escape(first["metric_name"])} '
                   f'불리한 평균 차이 {first["adverse_difference"]:+.3f}점. '
                   '이는 집단 차이에 대한 추가 진단 후보로, 정책 효과나 원인을 입증하지 않습니다.')
    else:
        screening='기본 우선 진단 기준을 충족한 집단은 확인되지 않았습니다.'
    notes=[
        ('RQ1 · 직무만족', satisfaction),
        ('RQ2 · 이직의도', turnover),
        ('RQ3 · HR 우선 진단',screening + ' 선별 기준은 현장 진단 대상 확인용이며, 예산 투자·개선 요인의 우선순위는 05번의 3단계 판단 절차로 추가 확인합니다.'),
        ('추가 분석 · SEM 간접경로',
         f'조직신뢰 → 직무만족 → 이직의도 간접경로 H6는 {effect("H6")}였고, ' f'직접경로 H3와 합한 전체 경로 합은 {effect("trust_total")}였습니다. '
         '계획서의 H1~H5와 별개의 탐색적 보조 결과이며, 단일 시점 자료로 인과적 매개효과를 입증하지 않습니다.'),
    ]
    result=''.join('<div class="rq-answer"><div class="rq-answer-title">'+escape(title)+
                   '</div><div class="rq-answer-text">'+detail+'</div></div>' for title,detail in notes)
    limit=('<p class="rq-answer-limit"><b>통계적 한계:</b> SEM은 조직신뢰 잠재변수·의사소통 합성지표를 사용하는 '
           f'대안 혼합형 모형입니다. 두 구성개념의 강한 중복성과 SEM RMSEA {sem["fit"]["rmsea"]:.3f}, '
           'OLS와 SEM의 의사소통-직무만족 경로 불일치를 고려하면 특정 조직요인의 독립적 효과를 확정하기 어렵습니다. '
           'OR은 확률이나 OLS 점수 변화량이 아닙니다. 이직의도 모형에서는 직무만족의 경계별 계수동일성에 주의해야 합니다. 직무만족 순서형 모형의 비례오즈 가정은 별도 검정하지 않았습니다.</p>')
    return '<div class="rq-answer-wrap">'+result+limit+'</div>'



def mediation_path_summary_html(sem):
    """Display the SEM-specific path sum, using its separately bootstrapped CIs.

    H3 and H6 are SEM effects. The total is a structural path sum, NOT an
    identified causal total effect in this cross-sectional survey.
    """
    e = {row['key']: row for row in sem['effects']}
    assert abs(e['H1']['estimate'] * e['H5']['estimate'] - e['H6']['estimate']) < 1e-9
    assert abs(e['H3']['estimate'] + e['H6']['estimate'] - e['trust_total']['estimate']) < 1e-9
    paths = [
        ('직접경로 · H3', '조직신뢰 → 이직의도', e['H3']),
        ('간접경로 · H6', '조직신뢰 → 직무만족 → 이직의도', e['H6']),
        ('전체 경로 합', '직접경로 + 간접경로', e['trust_total']),
    ]
    rows = []
    for title, desc, effect in paths:
        rows.append(
            '<div class="mediation-row">'
            f'<div class="mediation-label">{escape(title)}</div>'
            f'<div class="mediation-descr">{escape(desc)}</div>'
            f'<div class="mediation-value">{effect["estimate"]:+.3f}</div>'
            f'<div class="mediation-ci">[{effect["ci_low"]:+.3f}, {effect["ci_high"]:+.3f}]</div>'
            '</div>'
        )
    h1, h5 = e['H1']['estimate'], e['H5']['estimate']
    direct, indirect, total = (e['H3']['estimate'],e['H6']['estimate'],e['trust_total']['estimate'])
    formula=(
        f'간접경로(H6) = 조직신뢰 → 직무만족(H1) {h1:+.3f} × '
        f'직무만족 → 이직의도(H5) {h5:+.3f} = {indirect:+.3f}. '
        f'전체 경로 합 = 직접경로 {direct:+.3f} + 간접경로 {indirect:+.3f} = {total:+.3f}.'
    )
    return (
        '<section class="mediation-breakdown" aria-label="조직신뢰 매개경로 상세 비교">'
        '<div class="mediation-row mediation-heading">'
        '<div>구분</div><div>경로</div><div>SEM B</div><div>기업 부트스트랩 95% CI</div>'
        '</div>'+''.join(rows)+'</section>'
        f'<p class="mediation-formula">{formula}</p>'
        '<p class="mediation-caveat">각 신뢰구간은 500회 기업 단위 재표집에서 별도로 산출했으며 '
        '신뢰구간끼리 더한 값이 아닙니다. 단면 관찰자료이므로 이는 가정된 모형의 '
        '<b>통계적 간접경로</b>이지 인과적 매개효과의 증거는 아닙니다.</p>'
    )


def hr_decision_framework_html(priority):
    """Separate statistical screening from verified intervention rationale.

    All input figures come from the fixed common sample in
    priority_validation_v3.json; no investment ROI or causal effect invented.
    """
    definition = priority['screening_definition']
    candidates = priority['selected_at_default']
    contrast_count = int(priority['contrasts'])
    count = len(candidates)
    candidate_detail = '기준을 충족한 집단이 없습니다. 탐색적 모니터링을 유지합니다.'
    if candidates:
        first = candidates[0]
        candidate_detail = (
            f'<b>{escape(first["group_name"])} 직무</b> · '
            f'{escape(first["metric_name"])} 불리한 차이 '
            f'<b>{first["adverse_difference"]:+.3f}점</b> · '
            f'응답자 <b>{int(first["n"]):,}명</b> / '
            f'<b>{int(first["firms_in_group"]):,}개 기업</b> · '
            f'BH 보정 q=<b>{first["q_bh_152"]:.4f}</b>'
        )
    coincident = priority.get('same_group_turnover_and_negative_factors_at_0_2', [])
    if coincident:
        diagnostic_text = (
            f'같은 집단에서 이직의도와 함께 불리하게 관찰된 지표가 {len(coincident)}건 있습니다. '
            '현장 자료와 추가 분석으로 재확인한 뒤 개입 가능성을 판단합니다.'
        )
    else:
        diagnostic_text = (
            '기준 충족 집단에서 <b>조직신뢰·의사소통·직무만족의 불리한 차이가 함께 '
            '확인된 요인은 없습니다</b>(차이 0.2점 완화 기준). '
            '따라서 특정 요인에 곧바로 투자할 근거는 아직 부족합니다.'
        )
    return (
        '<div class="hr-decision-grid" aria-label="HR 투자의 근거와 판단 단계">'
        '<section class="hr-decision-card">'
        '<div class="hr-decision-step">01 · 우선 진단 집단 선별</div>'
        '<h4>통계적 위험 신호 확인</h4>'
        f'<p>불리한 평균 차이 <b>{definition["adverse_gap_default"]:.1f}점 이상</b>, '
        '해당 집단·나머지 집단 각각 <b>50명·10개 기업 이상</b>, '
        f'기업 군집강건 95% CI 하한 &gt; 0, 전체 {contrast_count}개 비교 '
        f'BH 보정 <b>q &lt; {definition["q_bh_below"]:.2f}</b>.</p>'
        f'<div class="hr-decision-result">해당 {count}건 · {candidate_detail}</div>'
        '</section>'
        '<section class="hr-decision-card">'
        '<div class="hr-decision-step">02 · 개선 요인 확인</div>'
        '<h4>원인과 개입 가능성 재진단</h4>'
        f'<p>{diagnostic_text}</p>'
        '<div class="hr-decision-result">면담, 업무량·근로조건, 관리·소통 방식 등을 '
        '함께 점검합니다. 전체 표본 OLS·SEM 계수로 특정 집단의 개선 효과를 단정하지 않습니다.</div>'
        '</section>'
        '<section class="hr-decision-card">'
        '<div class="hr-decision-step">03 · 시범 투자 결정</div>'
        '<h4>실행·평가 조건 충족 시 파일럿</h4>'
        '<p>현장에서 확인한 개선 문제, 실행 가능한 조치, 담당자·비용 한도 및 '
        '사전 평가계획이 있을 때만 소규모 시범사업을 검토합니다.</p>'
        '<div class="hr-decision-result">시범 전후 직무만족·이직의도, 가능하면 '
        '실제 퇴사율과 비용을 측정하고 비교기준을 마련한 뒤 확대·보류를 결정합니다.</div>'
        '</section>'
        '</div>'
        '<p class="hr-decision-caveat">투자 판단 기준은 <b>통계적 진단 → 현장 검증 → 시범사업 평가</b>의 '
        '의사결정 절차입니다. 0.3점은 탐색적 집단 선별의 연구 규칙이며, '
        '투자 수익률이나 프로그램 효과를 보장하는 임계값이 아닙니다. '
        '02번과 05번의 선별 결과는 사이드바 필터와 무관한 전체 공통 표본 기준입니다.</p>'
    )
