# -*- coding:utf-8 -*-
import numpy as np
from movie_item import moviecontent
from movie_data import dataset
from movie_item import moviename
from math import sqrt

#No.196の見たジャンルの和を取り、そのベクトルと映画データ全体との内積による類似度推薦
movierating = 0
movie = []
for movieID in dataset["196"]:
	movie.append(movieID)
	rating = dataset["196"][movieID]
	content = np.array(map(int,moviecontent[movieID]))
	movierating += rating*content

averating = movierating/len(dataset["196"])

value_of_ave = sqrt(np.dot(averating, averating))
sim_list = []

for m in moviecontent:
	if m in movie:
		continue
	else:
		c = np.array(map(int,moviecontent[m]))
		dot = np.dot(averating, c)
		cvalue = sqrt(np.dot(c,c))
		sim = dot/(cvalue*value_of_ave)
		sim_list.append([sim, m])

sim_list.sort()
sim_list.reverse()
print sim_list[:5]
print "No.196におすすめTop5は(content)"
for sim, movieID in sim_list[:5]:
	print moviename[movieID]



