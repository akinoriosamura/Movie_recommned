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

#item = [[userID, movieID, rating, timestamp]]
item = []
for line in open("./ml-100k/u.data", "r"):
	itemlist = line[:-1].split("\t")
	item.append(itemlist)

#mtrix_factorizationのためにユーザー✕映画の行列に格納し、値を修正、再度戻す
#１〜１６８２で並んだ（行の数：1の映画＋2の映画＋。。。）
for x in item:
	x[0] = int(x[0])
	x[1] = int(x[1])
item.sort()
item.reverse()
#行列に格納
data = np.zeros([943, 1682])
o = 0
for i in range(1,944):
	for j in range(o,len(item)):
		if item[j][0] == i:
			data[i-1][item[j][1]] = item[j][2]
			o += 1
		if item[j][0] > i:
			break

R = data
N = len(R)
M = len(R[0])
K = 100
P = np.random.randn(N, K)
Q = np.random.randn(M, K)
nP ,nQ = matrix_factorization(R, P, Q, K)
#print nP, nQ
nR = np.dot(nP, nQ.T)
print nR

"""
#dataset = {UsesrID:[{movieID:rateing},{}]},{UsesrID:[{movieID:rating},{}]},,,
dataset = {}
glob_ave = 0

for i in item:
	dataset.setdefault(i[0],{})
	dataset[i[0]][i[float(i[2])
	glob_ave += float(i[2])

glob_ave /= len(item)

#映画の上映時刻も考慮、①桁目✕0.01倍だけ増やす
dataset2 = {}

for i in item:
	dataset2.setdefault(i[0],{})
	dataset2[i[0]][i[1]]=[float(i[2])*(1+(0.01*(int(i[3])/(10**8)))), (int(i[3])/(10**8))]

dataset3 = []

for i in item:
	dataset3.append([int(i[3]), i[0], i[1], float(i[2])])
dataset3.sort()

#過去、評価する際にその時点での平均が高いほど余計に高く評価されてるから、O(n^2)
for i in range(len(dataset3)):
	j = 0
	then_ave_rating = 0
	if i !=0:
		for k in range(i-1):
			movie_number = dataset3[i][3]
			if movie_number in dataset3[k]:
				j += 1
				then_ave_rating += dataset3[k][3]
		if j != 0 and then_ave_rating != 0:
			then_ave_rating /= j
			a = (then_ave_rating-3)/10
			dataset3[i][3] -= a
"""
