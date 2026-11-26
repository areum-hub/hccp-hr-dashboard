"""Visual components for the HCCP II People Analytics dashboard.

Charts intentionally separate observed descriptive statistics from full-sample
model estimates. All values are loaded from the included project files.
"""
import numpy as np
import plotly.graph_objects as go

NAVY = '#18181B'
BLUE = '#2563EB'
TEAL = '#0D9488'
CORAL = '#DC2626'
LIGHT = '#A1A1AA'
GRID = '#E4E4E7'
INK = '#3F3F46'
METRIC_COLORS = {
    'trust': BLUE,
    'comm': TEAL,
    'satisfaction': '#7C3AED',
    'turnover': CORAL,
}


def frame(fig, height=350, left=15, right=24, bottom=32, top=25, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=left, r=right, b=bottom, t=top),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Noto Sans KR, Malgun Gothic, sans-serif', size=12, color=INK),
        showlegend=legend, hoverlabel=dict(bgcolor='white', font_size=12),
        legend=dict(orientation='h', y=1.13, x=0, xanchor='left'),
    )
    return fig


def overview_levels(current, overall, labels, show_reference=True):
    keys = list(labels)
    names = [labels[k] for k in keys]
    fig = go.Figure()
    if show_reference:
        for key in keys:
            x1 = float(overall[key].mean())
            x2 = float(current[key].mean())
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[labels[key], labels[key]], mode='lines',
                line=dict(color='#CCD6E1', width=3), showlegend=False,
                hoverinfo='skip',
            ))
        fig.add_trace(go.Scatter(
            x=[float(overall[k].mean()) for k in keys], y=names,
            mode='markers', name='전체 자료 평균',
            marker=dict(color='white', size=15, symbol='circle', line=dict(color='#91A3B4', width=2)),
            hovertemplate='%{y}<br>전체 자료 평균 %{x:.2f}<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=[float(current[k].mean()) for k in keys], y=names,
        mode='markers+text', name='현재 선택 평균',
        marker=dict(color=[METRIC_COLORS[k] for k in keys], size=16),
        text=[f'{current[k].mean():.2f}' for k in keys], textposition='middle right',
        textfont=dict(size=12, color=NAVY),
        hovertemplate='%{y}<br>현재 선택 평균 %{x:.2f}<extra></extra>',
    ))
    fig.update_xaxes(range=[1, 5.38], tickvals=[1, 2, 3, 4, 5], title='5점 척도 평균',
                     gridcolor=GRID, zeroline=False)
    fig.update_yaxes(categoryorder='array', categoryarray=list(reversed(names)),
                     showgrid=False, title=None)
    return frame(fig, height=330, left=30, right=40, top=39 if show_reference else 20, legend=show_reference)




def observed_associations(data):
    """All six unadjusted Pearson relationships; cluster bootstrap 95% CI."""
    text = {
        '조직신뢰 ↔ 의사소통': '신뢰 ↔ 소통',
        '조직신뢰 ↔ 직무만족': '신뢰 ↔ 만족',
        '의사소통 ↔ 직무만족': '소통 ↔ 만족',
        '조직신뢰 ↔ 이직 고려 의향': '신뢰 ↔ 이직 고려',
        '의사소통 ↔ 이직 고려 의향': '소통 ↔ 이직 고려',
        '직무만족 ↔ 이직 고려 의향': '만족 ↔ 이직 고려',
    }
    d = data.copy()
    d['display'] = d['label'].map(text)
    fig = go.Figure(go.Scatter(
        x=d['r'], y=d['display'], mode='markers+text',
        marker=dict(size=12, color=[BLUE if r>=0 else TEAL for r in d['r']]),
        text=[f'{r:+.3f}' for r in d['r']], textposition='middle right',
        textfont=dict(size=11, color=NAVY),
        error_x=dict(type='data', symmetric=False,
                     array=(d['ci_high']-d['r']).clip(lower=0),
                     arrayminus=(d['r']-d['ci_low']).clip(lower=0),
                     color='#94A3B8', thickness=1.6),
        customdata=d[['label','ci_low','ci_high','n']].to_numpy(),
        hovertemplate='%{customdata[0]}<br>관찰 상관 r=%{x:+.3f}<br>95% 구간 [%{customdata[1]:+.3f}, %{customdata[2]:+.3f}]<br>표본 %{customdata[3]:,}명<extra></extra>',
    ))
    fig.add_vline(x=0, line=dict(color='#94A3B8', width=1.3, dash='dash'))
    fig.update_xaxes(title='피어슨 상관계수 r (단순 상관)', range=[-.5,1.30],
                     tickvals=[-.5,0,.5,1],gridcolor=GRID,zeroline=False)
    fig.update_yaxes(categoryorder='array', categoryarray=list(reversed(d['display'].tolist())),
                     showgrid=False,title=None, tickfont=dict(size=12))
    return frame(fig, height=360, left=22, right=38, top=15, bottom=40)


def score_histogram(df, key, name):
    fig = go.Figure(go.Histogram(
        x=df[key].dropna(), nbinsx=22, marker_color=METRIC_COLORS[key],
        hovertemplate='점수 구간: %{x}<br>응답: %{y:,}명<extra></extra>',
    ))
    fig.update_layout(bargap=0.08)
    fig.update_xaxes(title=f'{name} (1~5점)', range=[0.9, 5.1], gridcolor=GRID)
    fig.update_yaxes(title='응답자 수', gridcolor=GRID)
    return frame(fig, height=305)


def group_means(g, metric, reference, title, height=None):
    g = g.sort_values('평균').copy()
    if height is None:
        height = max(290, min(940, 44 * len(g) + 90))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g['평균'], y=g['집단'], mode='markers+text', name='집단 평균',
        marker=dict(color=METRIC_COLORS[metric], size=12),
        text=g['평균'].map(lambda v: f'{v:.2f}'), textposition='middle right',
        customdata=g[['응답자 수']].to_numpy(),
        hovertemplate='%{y}<br>평균 %{x:.2f}점<br>응답자 %{customdata[0]:,}명<extra></extra>',
    ))
    fig.add_vline(x=reference, line=dict(color='#8497A8', width=1.5, dash='dash'))
    fig.update_xaxes(title='평균 점수 (5점 척도)', range=[1, 5.39],
                     tickvals=[1, 2, 3, 4, 5], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(categoryorder='array', categoryarray=list(reversed(g['집단'].tolist())), title=None,
                     showgrid=False, tickfont=dict(size=11))
    fig.add_annotation(x=0.99, y=1.08, xref='paper', yref='paper',
                       text=f'점선: 현재 선택 표본 평균 {reference:.2f}', showarrow=False,
                       font=dict(size=11, color='#697C8E'), xanchor='right')
    return frame(fig, height=height, left=16, right=30, top=43)


def adverse_intervals(table, metric, cutoff):
    if table.empty:
        return go.Figure()
    show = table.sort_values('불리한 차이', ascending=False).head(15).copy()
    x = show['불리한 차이'].to_numpy(dtype=float)
    lower = show['불리한 구간 하한'].to_numpy(dtype=float)
    upper = show['불리한 구간 상한'].to_numpy(dtype=float)
    fig = go.Figure(go.Scatter(
        x=x, y=show['집단'], mode='markers',
        marker=dict(color=[CORAL if s == '우선 검토' else BLUE for s in show['판정']], size=11),
        error_x=dict(type='data', symmetric=False,
                     array=np.maximum(0, upper - x), arrayminus=np.maximum(0, x - lower),
                     color='#8399AE', thickness=1.3),
        customdata=show[['응답자 수','판정']].to_numpy(),
        hovertemplate='%{y}<br>불리한 차이: %{x:+.3f}점<br>응답: %{customdata[0]:,}명<br>%{customdata[1]}<extra></extra>',
    ))
    fig.add_vline(x=cutoff, line=dict(color=CORAL, width=1.3, dash='dot'))
    fig.add_vline(x=0, line=dict(color='#8796A6', width=1.2, dash='dash'))
    xmax = max(float(np.nanmax(upper)), cutoff, 0.2) + 0.11
    xmin = min(float(np.nanmin(lower)), 0) - 0.1
    fig.update_xaxes(title='불리한 방향의 집단 − 나머지 평균 차이 (점)',
                     range=[xmin, xmax], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(categoryorder='array', categoryarray=list(reversed(show['집단'].tolist())), showgrid=False)
    fig.add_annotation(text=f'판단 기준 {cutoff:.1f}점', x=cutoff, y=1.075, xref='x', yref='paper',
                       showarrow=False, font=dict(color=CORAL, size=11))
    return frame(fig, height=max(295, 40 * len(show) + 95), top=42)


def sensitivity_matrix(report):
    gaps = [.2, .3, .4]
    minimums = [30, 50, 100]
    z = [[int(report['sensitivity_counts'][f'g{g}_n{n}']) for n in minimums] for g in gaps]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f'{n}명' for n in minimums], y=[f'{g:.1f}점' for g in gaps],
        colorscale=[[0, '#EFF6FF'], [0.45, '#BFDBFE'], [1, '#2563EB']],
        zmin=0, zmax=max(5, max(map(max, z))), showscale=False,
        hovertemplate='차이 기준 %{y}<br>최소 인원 %{x}<br>선별 조합 %{z}건<extra></extra>',
    ))
    for i, g in enumerate(gaps):
        for j, n in enumerate(minimums):
            val = z[i][j]
            fig.add_annotation(x=f'{n}명', y=f'{g:.1f}점', text=f'<b>{val}건</b>',
                               showarrow=False, font=dict(size=17, color='white' if val>=4 else NAVY))
    fig.update_xaxes(title='최소 집단 인원', side='bottom')
    fig.update_yaxes(title='평균 차이 기준', autorange='reversed')
    return frame(fig, height=265, left=20, right=12, top=15)


def item_averages(df, mappings, color):
    rows=[(label, float(df[code].mean()), int(df[code].count())) for code,label in mappings]
    rows.sort(key=lambda x:x[1])
    fig=go.Figure(go.Bar(
        x=[v[1] for v in rows], y=[v[0] for v in rows], orientation='h',
        marker_color=color, text=[f'{v[1]:.2f}' for v in rows], textposition='outside',
        customdata=[[v[2]] for v in rows],
        hovertemplate='%{y}<br>평균 %{x:.2f}<br>응답자 %{customdata[0]:,}명<extra></extra>',
    ))
    fig.update_xaxes(title='평균 점수 (1~5점)', range=[0, 5.35],
                     tickvals=[1,2,3,4,5], gridcolor=GRID)
    fig.update_yaxes(title=None, gridcolor='rgba(0,0,0,0)')
    return frame(fig, height=335, left=15, right=35, top=5)


def regression_change(coefs):
    order=['단순 의사소통','조직신뢰 추가','직무만족 추가','전체 통제']
    names=['의사소통만 분석','조직신뢰 함께 고려','직무만족까지 고려','개인·직무·기업 특성 통제']
    g=coefs[(coefs['variable']=='comm') & coefs['model'].isin(order)].copy()
    g['order']=g['model'].map({v:i for i,v in enumerate(order)})
    g=g.sort_values('order')
    fig=go.Figure(go.Scatter(
        x=g['coef'], y=names, mode='markers+text',
        marker=dict(color=[BLUE if v<0 else CORAL for v in g['coef']], size=13),
        text=[f'{v:+.3f}' for v in g['coef']], textposition='top center',
        error_x=dict(type='data', symmetric=False,
                     array=(g['ci_high']-g['coef']).to_numpy(),
                     arrayminus=(g['coef']-g['ci_low']).to_numpy(),
                     color='#788FA5', thickness=1.8),
        hovertemplate='%{y}<br>회귀계수 %{x:+.3f}<extra></extra>',
    ))
    fig.add_vline(x=0, line=dict(color='#8093A4', width=1.3, dash='dash'))
    fig.update_xaxes(title='의사소통 회귀계수 B (95% 신뢰구간)',
                     range=[-0.47,0.25], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(categoryorder='array', categoryarray=list(reversed(names)),
                     showgrid=False, title=None)
    return frame(fig, height=360, left=12, right=26, top=23)


def regression_forest(table, var_labels):
    g=table.copy()
    g['항목']=g['variable'].map(var_labels).fillna(g['variable'])
    g=g.sort_values('coef')
    fig=go.Figure(go.Scatter(
        x=g['coef'], y=g['항목'], mode='markers',
        marker=dict(color=[BLUE if v<0 else CORAL for v in g['coef']],size=12),
        error_x=dict(type='data', symmetric=False,
                     array=(g['ci_high']-g['coef']).to_numpy(),
                     arrayminus=(g['coef']-g['ci_low']).to_numpy(),
                     color='#899CAD',thickness=1.7),
        hovertemplate='%{y}<br>회귀계수 %{x:+.3f}<extra></extra>',
    ))
    fig.add_vline(x=0,line=dict(color='#8598A9',dash='dash'))
    fig.update_xaxes(title='회귀계수 B (95% 신뢰구간)',gridcolor=GRID,zeroline=False)
    fig.update_yaxes(categoryorder='array',categoryarray=list(reversed(g['항목'].tolist())),showgrid=False)
    return frame(fig,height=max(275,52*len(g)+90),top=18)


def path_diagram(m):
    fig=go.Figure()
    nodes=[(.015,.255,'조직신뢰','X'),(.375,.615,'직무만족','M'),(.735,.985,'이직 고려 의향','Y')]
    for xmin,xmax,label,sub in nodes:
        fig.add_shape(type='rect',x0=xmin,x1=xmax,y0=.53,y1=.81,
                      line=dict(color='#A6BCD0',width=1.5),fillcolor='#F1F7FD')
        fig.add_annotation(x=(xmin+xmax)/2,y=.68,text=f'<b>{label}</b><br><span style="font-size:11px">{sub}</span>',
                           showarrow=False,font=dict(size=16,color=NAVY),align='center')
    fig.add_annotation(x=.366,y=.67,ax=.263,ay=.67,xref='x',yref='y',axref='x',ayref='y',
                       text='',showarrow=True,arrowhead=2,arrowsize=1.2,arrowwidth=2.2,arrowcolor=BLUE)
    fig.add_annotation(x=.724,y=.67,ax=.624,ay=.67,xref='x',yref='y',axref='x',ayref='y',
                       text='',showarrow=True,arrowhead=2,arrowsize=1.2,arrowwidth=2.2,arrowcolor=BLUE)
    fig.add_annotation(x=.313,y=.93,text=f'a = {m["path_a"]:+.3f}',showarrow=False,font=dict(color=NAVY,size=15))
    fig.add_annotation(x=.677,y=.93,text=f'b = {m["path_b"]:+.3f}',showarrow=False,font=dict(color=NAVY,size=15))
    fig.add_shape(type='path',path='M 0.15 0.52 L 0.15 0.30 L 0.855 0.30 L 0.855 0.52',
                  line=dict(color=CORAL,width=2,dash='dot'))
    fig.add_annotation(x=.855,y=.53,ax=.855,ay=.34,xref='x',yref='y',axref='x',ayref='y',
                       text='',showarrow=True,arrowhead=2,arrowsize=1.2,arrowwidth=2,arrowcolor=CORAL)
    fig.add_annotation(x=.5,y=.225,text=f'직접경로 c′ = {m["direct"]:+.3f}',
                       showarrow=False,font=dict(size=14,color=CORAL))
    fig.update_xaxes(visible=False,range=[0,1],fixedrange=True)
    fig.update_yaxes(visible=False,range=[0,1.11],fixedrange=True)
    return frame(fig,height=280,left=7,right=7,bottom=0,top=2)


def indirect_forest(m):
    vals=[(140,m['stability']['first_140_ci']),
          (500,m['stability']['first_500_ci']),
          (1000,m['stability']['all_1000_ci'])]
    names=[f'{n:,}회' for n,_ in vals]
    center=[sum(v)/2 for _,v in vals]
    low=[v[0] for _,v in vals]
    high=[v[1] for _,v in vals]
    fig=go.Figure(go.Scatter(
        x=center,y=names,mode='markers',
        marker=dict(color=[LIGHT,LIGHT,TEAL],size=[10,10,14]),
        error_x=dict(type='data',symmetric=False,
                     array=[h-c for h,c in zip(high,center)],
                     arrayminus=[c-l for c,l in zip(center,low)],
                     color='#5A8295',width=5,thickness=2),
        customdata=[[l,h] for l,h in zip(low,high)],
        hovertemplate='반복 횟수 %{y}<br>95% 구간 [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>',
    ))
    fig.add_vline(x=0,line=dict(color='#8494A3',dash='dash'))
    fig.update_xaxes(title='간접관계 95% 백분위 신뢰구간',range=[-.205,.045],
                     gridcolor=GRID,zeroline=False)
    fig.update_yaxes(categoryorder='array',categoryarray=list(reversed(names)),showgrid=False,title=None)
    return frame(fig,height=268,left=10,right=25,top=17)


def indirect_point(m):
    fig=go.Figure(go.Scatter(
        x=[m['indirect']],y=['간접관계'],mode='markers',
        marker=dict(size=15,color=TEAL),
        error_x=dict(type='data',symmetric=False,
                     array=[m['indirect_ci_high']-m['indirect']],
                     arrayminus=[m['indirect']-m['indirect_ci_low']],
                     color='#467A8C',width=6,thickness=2),
        hovertemplate='간접관계 추정값 %{x:+.3f}<extra></extra>',
    ))
    fig.add_vline(x=0,line=dict(color='#9CAAB6',dash='dash'))
    fig.update_xaxes(title='a × b',range=[-.21,.045],gridcolor=GRID,zeroline=False)
    fig.update_yaxes(showticklabels=False,showgrid=False,zeroline=False)
    return frame(fig,height=180,left=0,right=14,top=0,bottom=35)


def priority_forest(table, screening, threshold):
    """Full-sample cluster-adjusted 95% group-vs-rest contrasts (top 12 by adverse gap)."""
    show=table.loc[(table['n']>=30)&(table['firms_in_group']>=10)].copy()
    show['screening']=screening.reindex(show.index,fill_value=False)
    show=show.sort_values('adverse_difference',ascending=False).head(12)
    if show.empty: return go.Figure()
    show['label']=show['dimension']+' · '+show['group_name']+' / '+show['metric_name']
    x=show['adverse_difference'].to_numpy(dtype=float)
    lo=show['adverse_ci_low'].to_numpy(dtype=float)
    hi=show['adverse_ci_high'].to_numpy(dtype=float)
    colors=[CORAL if v else BLUE for v in show['screening']]
    fig=go.Figure(go.Scatter(x=x,y=show['label'],mode='markers',
        marker=dict(color=colors,size=[13 if z else 9 for z in show['screening']]),
        error_x=dict(type='data',symmetric=False,
            array=np.maximum(0,hi-x),arrayminus=np.maximum(0,x-lo),
            color='#839AAF',thickness=1.5),
        customdata=show[['n','firms_in_group','q_bh_152']].to_numpy(),
        hovertemplate='%{y}<br>불리한 차이 %{x:+.3f}점<br>응답자 %{customdata[0]:,}명 / 기업 %{customdata[1]:,}개<br>BH 보정 q=%{customdata[2]:.4f}<extra></extra>'))
    fig.add_vline(x=threshold,line=dict(color=CORAL,dash='dot',width=1.7))
    fig.add_vline(x=0,line=dict(color='#8493A4',dash='dash',width=1.2))
    fig.update_xaxes(title='불리한 방향의 평균 차이 (점) · 기업 군집강건 95% 신뢰구간',
                     gridcolor=GRID,zeroline=False)
    fig.update_yaxes(title=None,categoryorder='array',
                     categoryarray=list(reversed(show['label'].tolist())),showgrid=False,tickfont=dict(size=11))
    return frame(fig,height=495,left=5,right=20,top=20,bottom=45)


def sem_hypothesis():
    """Proposed SEM paths H1-H5; the H6 indirect path is H1 x H5.

    Plot labels show hypotheses/directions, never fitted SEM coefficients.
    """
    fig=go.Figure()
    nodes=[
        (.030,.270,.70,.88,'조직신뢰','잠재요인 후보 · 3문항',BLUE),
        (.030,.270,.22,.40,'의사소통','잠재요인 후보 · 4문항',TEAL),
        (.450,.660,.46,.64,'직무만족','관측변수 · 1문항',BLUE),
        (.795,.985,.46,.64,'이직 고려 의향','관측변수 · 1문항',CORAL),
    ]
    for x0,x1,y0,y1,title,subtitle,color in nodes:
        fig.add_shape(type='rect', x0=x0,x1=x1,y0=y0,y1=y1,
                      line=dict(color=color,width=1.5),fillcolor='#F2F7FA',
                      layer='below')
        fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2+.042,text=f'<b>{title}</b>',
                           showarrow=False,font=dict(size=13,color=NAVY))
        fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2-.053,text=subtitle,
                           showarrow=False,font=dict(size=10,color='#597185'))
    def path_arrow(x0,y0,x1,y1,color,width=2.1):
        fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref='x',yref='y',
                           axref='x',ayref='y',text='',showarrow=True,
                           arrowhead=2,arrowsize=1.15,arrowwidth=width,arrowcolor=color)
    # Principal paths to/from the mediator.
    path_arrow(.273,.790,.448,.599,BLUE)
    path_arrow(.273,.310,.448,.506,TEAL)
    path_arrow(.663,.550,.792,.550,CORAL)
    # Dotted direct paths are conditional on the mediator in the SEM model.
    fig.add_trace(go.Scatter(x=[.245,.37,.875,.875],y=[.882,.963,.963,.757],
        mode='lines',line=dict(color=BLUE,width=1.75,dash='dot'),
        hoverinfo='skip',showlegend=False))
    path_arrow(.875,.757,.875,.645,BLUE,1.75)
    fig.add_trace(go.Scatter(x=[.245,.37,.922,.922],y=[.216,.094,.094,.352],
        mode='lines',line=dict(color=TEAL,width=1.75,dash='dot'),
        hoverinfo='skip',showlegend=False))
    path_arrow(.922,.352,.922,.457,TEAL,1.75)
    # Exogenous covariance is not a causal SEM arrow.
    fig.add_shape(type='line',x0=.15,x1=.15,y0=.403,y1=.696,
                  line=dict(color='#91A6B6',width=1.5,dash='dash'))
    fig.add_annotation(x=.16,y=.543, xanchor='left',
                       text='공분산 허용',showarrow=False,
                       font=dict(size=10,color='#65798C'))
    annotations=[
        (.366,.753,'<b>H1 ＋</b>',BLUE),
        (.359,.359,'<b>H2 ＋</b>',TEAL),
        (.731,.620,'<b>H5 －</b>',CORAL),
        (.580,.924,'<b>H3 －</b>  직접경로',BLUE),
        (.582,.133,'<b>H4 －</b>  직접경로',TEAL),
    ]
    for x,y,label,color in annotations:
        fig.add_annotation(x=x,y=y,text=label,showarrow=False,
                           font=dict(size=12,color=color))
    fig.update_xaxes(visible=False,range=[0,1.03],fixedrange=True)
    fig.update_yaxes(visible=False,range=[0,1.04],fixedrange=True)
    return frame(fig,height=355,left=7,right=7,top=2,bottom=2)


def ordinal_odds_forest(findings):
    """Odds ratios from a clustered 5-category proportional-odds model."""
    labels = [r['label'] for r in findings]
    point = np.asarray([r['odds_ratio'] for r in findings], dtype=float)
    low = np.asarray([r['or_ci_low'] for r in findings], dtype=float)
    high = np.asarray([r['or_ci_high'] for r in findings], dtype=float)
    colors = [BLUE if r['ordinal_coef'] < 0 else CORAL for r in findings]
    fig = go.Figure(go.Scatter(
        x=point, y=labels, mode='markers+text',
        text=[f'{x:.3f}' for x in point], textposition='top center',
        marker=dict(color=colors, size=13),
        error_x=dict(type='data', symmetric=False, array=high-point,
                     arrayminus=point-low, color='#7D93A7',thickness=1.8),
        customdata=np.column_stack([low, high]),
        hovertemplate='%{y}<br>오즈비 %{x:.3f}<br>기업 군집 95% CI [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>',
    ))
    fig.add_vline(x=1,line=dict(color='#93A1B0',dash='dash',width=1.4))
    fig.update_xaxes(title='이직 고려 의향의 높은 응답범주에 대한 누적 오즈비 (OR)',
                     range=[0.30, 1.53],gridcolor=GRID,zeroline=False)
    fig.update_yaxes(categoryorder='array',categoryarray=list(reversed(labels)),
                     showgrid=False,title=None)
    fig.add_annotation(x=1,y=1.10,xref='x',yref='paper',text='OR = 1',
                       showarrow=False,font=dict(size=11,color='#708296'))
    return frame(fig,height=285,left=15,right=20,top=38,bottom=42)


def ordinal_threshold_patterns(rows):
    """Four separate cumulative binary logits, with cluster-adjusted intervals."""
    fig = go.Figure()
    order = [('trust','조직신뢰',BLUE), ('comm','의사소통',TEAL),
             ('satisfaction','직무만족',CORAL)]
    for key,label,color in order:
        sub=rows.loc[rows['key']==key].sort_values('cut')
        val=sub['coef'].to_numpy(dtype=float)
        lo=sub['ci_low'].to_numpy(dtype=float)
        hi=sub['ci_high'].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(
            x=sub['cut'].tolist(),y=val,mode='lines+markers',name=label,
            line=dict(color=color,width=2),marker=dict(size=10,color=color),
            error_y=dict(type='data',symmetric=False,array=hi-val,arrayminus=val-lo,
                         color=color,thickness=1.2,width=3),
            customdata=np.column_stack([lo,hi,sub['n_event'].to_numpy()]),
            hovertemplate='%{fullData.name}<br>이직 고려 %{x}점 초과<br>로짓계수 %{y:+.3f}<br>95% CI [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<br>해당범주 응답 %{customdata[2]:,.0f}명<extra></extra>',
        ))
    fig.add_hline(y=0,line=dict(color='#91A2B1',dash='dash',width=1.2))
    fig.update_xaxes(tickmode='array',tickvals=[1,2,3,4],
                     ticktext=['1점 초과','2점 초과','3점 초과','4점 초과'],
                     title='이직 고려 의향 응답 경계',gridcolor=GRID)
    fig.update_yaxes(title='경계별 이진로짓 계수 (95% 군집강건 구간)',
                     gridcolor=GRID,zeroline=False)
    return frame(fig,height=345,left=22,right=25,top=48,bottom=40,legend=True)


def sem_fitted_paths(sem_model):
    """Actual fitted hybrid SEM structural paths (not an unfitted hypothesis diagram)."""
    effects={e['key']:e for e in sem_model['effects']}
    fig=go.Figure()
    nodes=[
        (.035,.275,.685,.875,'조직신뢰','잠재변수 · 3문항',BLUE),
        (.035,.275,.225,.415,'의사소통','관측 합성지표 · 4문항',TEAL),
        (.455,.675,.455,.635,'직무만족','관측변수 · 1문항',BLUE),
        (.795,.995,.455,.635,'이직 고려 의향','관측변수 · 1문항',CORAL),
    ]
    for x0,x1,y0,y1,name,kind,color in nodes:
        fig.add_shape(type='rect',x0=x0,x1=x1,y0=y0,y1=y1,
                      line=dict(color=color,width=1.8),fillcolor='#F2F7FA',layer='below')
        fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2+.032,text=f'<b>{name}</b>',
                           showarrow=False,font=dict(size=13,color=NAVY))
        fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2-.053,text=kind,
                           showarrow=False,font=dict(size=10,color='#587184'))
    def arrow(a,b,c,d,color):
        fig.add_annotation(x=c,y=d,ax=a,ay=b,xref='x',yref='y',axref='x',ayref='y',
                           text='',showarrow=True,arrowhead=2,arrowsize=1.1,
                           arrowwidth=2.0,arrowcolor=color)
    arrow(.278,.777,.452,.6,BLUE)
    arrow(.278,.318,.452,.508,TEAL)
    arrow(.677,.55,.793,.55,CORAL)
    # Direct paths: dashed to visually differentiate from mediated ones.
    fig.add_trace(go.Scatter(x=[.25,.36,.885,.885],y=[.875,.955,.955,.757],mode='lines',
                             line=dict(color=BLUE,width=1.7,dash='dot'),showlegend=False,hoverinfo='skip'))
    arrow(.885,.758,.885,.636,BLUE)
    fig.add_trace(go.Scatter(x=[.255,.365,.936,.936],y=[.227,.11,.11,.349],mode='lines',
                             line=dict(color=TEAL,width=1.7,dash='dot'),showlegend=False,hoverinfo='skip'))
    arrow(.936,.35,.936,.459,TEAL)
    fig.add_shape(type='line',x0=.160,x1=.160,y0=.43,y1=.674,
                  line=dict(color='#8A9DAD',width=1.5,dash='dash'))
    fig.add_annotation(x=.174,y=.55,xanchor='left',text='상관 허용',showarrow=False,
                       font=dict(size=9,color='#697E8D'))
    for xpos,ypos,key,color in [
        (.366,.75,'H1',BLUE),(.356,.354,'H2',TEAL),(.736,.612,'H5',CORAL),
        (.590,.920,'H3',BLUE),(.587,.145,'H4',TEAL),
    ]:
        fig.add_annotation(x=xpos,y=ypos,text=f'<b>{key} {effects[key]["estimate"]:+.3f}</b>',
                           showarrow=False,font=dict(size=12,color=color))
    fig.update_xaxes(visible=False,range=[0,1.04],fixedrange=True)
    fig.update_yaxes(visible=False,range=[0,1.05],fixedrange=True)
    return frame(fig,height=355,left=8,right=8,top=2,bottom=2)


def sem_effect_forest(effects):
    """Bootstrap 95% CI of actually estimated SEM structural coefficients."""
    ordered=[next(e for e in effects if e['key']==k) for k in ['H1','H2','H3','H4','H5','H6']]
    names=[f'{e["key"]} · {e["label"]}' for e in ordered]
    v=np.array([e['estimate'] for e in ordered]);l=np.array([e['ci_low'] for e in ordered]);h=np.array([e['ci_high'] for e in ordered])
    colors=[TEAL if e['ci_low']>0 else BLUE if e['ci_high']<0 else '#9CA6B3' for e in ordered]
    fig=go.Figure(go.Scatter(x=v,y=names,mode='markers+text',
            text=[f'{x:+.3f}' for x in v],textposition='top center',
            marker=dict(size=12,color=colors),
            error_x=dict(type='data',symmetric=False,array=h-v,arrayminus=v-l,
                         color='#8B9AA9',thickness=1.8),
            hovertemplate='%{y}<br>경로 추정값 %{x:+.3f}<extra></extra>',showlegend=False))
    fig.add_vline(x=0,line=dict(color='#A0AFBD',dash='dash',width=1.5))
    fig.update_xaxes(title='SEM 경로계수 · 95% 기업 재표집 구간 (H6은 경로계수의 곱)',
                     zeroline=False,gridcolor=GRID)
    fig.update_yaxes(categoryorder='array',categoryarray=list(reversed(names)),showgrid=False,title=None)
    return frame(fig,height=400,left=5,right=20,top=18,bottom=50)
