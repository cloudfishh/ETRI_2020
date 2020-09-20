"""
Accumulation detection with probabilistic forecast

2020. 09. 15. Tue.
Soyeong Park
"""

'''
1. household 1개 데이터 불러오기
2. injection 하기
3. accumulation detection
    - probabilistic forecast로 판단하기?
    - candidate(before value)에 대해 prob. forecast를 하고 그 candidate가 위치한 interval에 따라 (z-score) 판단.
    - criteria? hmm?
    - 그럼 일단 다음주까진 candidate의 z-score 분석
4. imputation
5. result
'''
