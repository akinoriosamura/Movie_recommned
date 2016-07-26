#! /usr/bin/env python                                                          
# -*- coding: utf-8 -*-                                                         

from numpy import *

costemer = array([[1,2,1,0]])

cv = matrix([[0,1,1,0,1],
	         [0,0,4,0,2],
	         [1,0,3,0,1],
	         [3,1,0,0,1]
	         ])

ip = dot(costemer, cv)
shop = ["shop1","shop2","shop3","shop4","shop5"]
z = zip(shop, ip.tolist()[0])
print z
result = sorted(z, key=lambda x:x[1],reverse=True)

print "おすすめは...."
for i in range(len(result)):
	print "第",str(i+1),"位",result[i][0],"：","コサインによる類似度",result[i][1]