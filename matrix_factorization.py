# -*- coding:utf-8 -*-

import numpy as np

def matrix_factorization(R, P, Q, K, alpha = 0.0002, beta = 0.02):
	Q = Q.T
	for step in range(10):
		#正規化なし
		"""
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i,:], Q[:,j])
					for k in range(K):
						P[i][k] = P[i][k] + 2*alpha*eij*Q[k][j]
						Q[k][j] = Q[k][j] + 2*alpha*eij*P[i][k]
		"""

		eij = 0.0
		e = 0.0
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in range(K):
						eij += (beta/2)*(pow(P[i][k], 2) + pow(Q[k][j], 2))
					for k in range(K):
						P[i][k] = P[i][k] + alpha*(2*eij*Q[k][j] - beta*P[i][k])
						Q[k][j] = Q[k][j] + alpha*(2*eij*P[i][k] - beta*Q[k][j])
					e += eij
		if e < 0.001:
			break
		print step
	return P, Q.T


R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]


R = np.array(R)
print R
N = len(R)
M = len(R[0])

K = 2

P = np.random.randn(N, K)
Q = np.random.randn(M, K)

nP ,nQ = matrix_factorization(R, P, Q, K)
print nP, nQ
nR = np.dot(nP, nQ.T)
print nR
