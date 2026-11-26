import pandas as pd
labels={'W23IND1':'산업','W23SCALE':'기업규모','W23Q03B':'직급','W23Q04':'직무','W23DQ01':'성별'}
# HCCPⅡ 4차 공식 근로자 레이아웃의 응답범주를 표시용 명칭으로 변환합니다.
# 원본 수치 코드는 분석 데이터에 그대로 유지됩니다.
CATEGORY_NAMES={
 'W23IND1':{1:'제조',2:'금융',3:'비금융'},
 'W23SCALE':{1:'100~299인',2:'300~999인',3:'1,000인 이상'},
 'W23Q03B':{1:'사원급',2:'주임/계장',3:'대리급',4:'과장급',5:'차장급',6:'부장급',7:'임원급',8:'생산직 반장급'},
 'W23Q04':{1:'연구개발',2:'연구개발 기획·관리',3:'국내영업',4:'해외영업',5:'대고객지원',6:'A/S',7:'생산관리',8:'생산기술',9:'품질관리',10:'전략기획',11:'인사(인사·교육·노사)',12:'재무·회계·경리',13:'전산(IT)',14:'PR/IR(홍보)',15:'법무·총무',16:'구매·자재',17:'마케팅(영업기획)',18:'환경안전',19:'품질검사',20:'생산(제조)',21:'금융상품개발',22:'자금운용 및 투자',23:'서비스직',24:'핵심전문직',25:'기타'},
 'W23DQ01':{1:'남자',2:'여자'},
}
def category_name(column,code):
    if pd.isna(code): return '미응답'
    key=int(code)
    return CATEGORY_NAMES[column].get(key,f'미확인 코드 {key}')
metrics={'trust':'조직신뢰','comm':'의사소통','satisfaction':'직무만족','turnover':'이직 고려 의향'}
items={'W23Q22D':'조직 내 협동·신뢰','W23Q24D':'동료 신뢰','W23Q24F':'경영진 신뢰','W23Q22F':'조직원 간 의사소통','W23Q24A':'회사 정보공유','W23Q24B':'자유로운 의견개진','W23Q24C':'부서 간 소통'}


# 2023 HCCP II Worker 4th wave: the labels below reproduce the survey layout
# (leading survey item numbering omitted in the display only).
RESEARCH_VARIABLES = [
    {'name': '조직신뢰', 'role': '선행 설명변수 (X)',
     'measurement': '3문항 · 잠재변수(SEM)',
     'codes': ('W23Q22D', 'W23Q24D', 'W23Q24F')},
    {'name': '의사소통', 'role': '추가 설명변수',
     'measurement': '4문항 평균 · 관측 합성지표(SEM)',
     'codes': ('W23Q22F', 'W23Q24A', 'W23Q24B', 'W23Q24C')},
    {'name': '직무만족', 'role': '매개변수 (M)',
     'measurement': '1문항 · 관측변수',
     'codes': ('W23Q261',)},
    {'name': '이직 고려 의향', 'role': '결과변수 (Y)',
     'measurement': '1문항 · 관측변수',
     'codes': ('W23Q27A',)},
]

OFFICIAL_ITEM_TEXT = {
    'W23Q22D': '조직 내에 협동 및 신뢰관계가 구축되어 있다',
    'W23Q24D': '동료들 간에 서로 신뢰한다',
    'W23Q24F': '경영진은 모든 면에서 믿고 따라갈 만하다',
    'W23Q22F': '조직 내에서 조직원들 사이에 의사소통이 잘 이루어진다',
    'W23Q24A': '직원들에게 회사 사정을 소상하게 알려준다',
    'W23Q24B': '상급자에게 의견을 자유롭게 낼 수 있다',
    'W23Q24C': '부서간의 커뮤니케이션이 잘 되는 편이다',
    'W23Q261': '하고 있는 일에 대한 전반적 만족도',
    'W23Q27A': '나는 조금이라도 좋은 조건을 제시하는 회사가 있으면 옮길 것을 고려해 보고 있다',
}

GROUP_VARIABLE_TEXT = {
    'W23IND1': '산업분류',
    'W23SCALE': '기업규모',
    'W23Q03B': '현재 직급',
    'W23Q04': '주 업무',
    'W23DQ01': '성별',
}


# H1-H5 retain the wording/direction of the original research plan.
# H6 is an extension specifically for the trust -> satisfaction -> turnover path.
# Hypotheses list; separately estimated SEM effects are stored in sem_structural.json.
SEM_HYPOTHESES = [
    {'id':'H1', 'path':'조직신뢰 → 직무만족', 'direction':'정(＋)', 'type':'직무만족 경로', 'source':'기존 가설'},
    {'id':'H2', 'path':'의사소통 → 직무만족', 'direction':'정(＋)', 'type':'직무만족 경로', 'source':'기존 가설'},
    {'id':'H3', 'path':'조직신뢰 → 이직 고려 의향', 'direction':'부(－)', 'type':'이직 고려 직접경로', 'source':'기존 가설'},
    {'id':'H4', 'path':'의사소통 → 이직 고려 의향', 'direction':'부(－)', 'type':'이직 고려 직접경로', 'source':'기존 가설'},
    {'id':'H5', 'path':'직무만족 → 이직 고려 의향', 'direction':'부(－)', 'type':'이직 고려 경로', 'source':'기존 가설'},
    {'id':'H6', 'path':'조직신뢰 → 직무만족 → 이직 고려 의향', 'direction':'간접(－)', 'type':'매개경로 · H1 × H5', 'source':'확장 가설'},
]
