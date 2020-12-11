import numpy as np
d = np.array([[3,2,1],
              [3,2,2],
              [5,3,-1],
              [-5,-2,-3],
              [-2,-1,2],
              [-6,-3,2],
              [1,0,2],
              [3,1,-3],
              [0,1,0],
              [-2,-1,-2]])

d_c = d-d.mean(axis=0)
cov = (1/9)*np.dot(d_c.T, d_c)
# cov_np = np.cov(d_c.T)    # 값 마자용~ 그냥 계수가 곱해졌을 뿐

a,b=np.linalg.eig(cov)
np.corrcoef(cov)