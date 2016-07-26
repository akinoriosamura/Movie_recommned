# -*- coding:utf-8 -*-
import numpy as np
from movie_data import dataset2
from movie_data import glob_ave
from movie_item import moviecontent
from movie_item import moviename
from movie_user import userdata
from math import sqrt,fabs

#ユーザー固有の正規化（辛口はよりマイルドに一般化）
""
def normalized_user(bar = None):
	for user in dataset2:
		user_ave = 0
		for movieID in dataset2[user]:
			user_ave += dataset2[user][movieID][0]
		normalized_value = (len(dataset2[user])/user_ave)/glob_ave
		for movieID in dataset2[user]:
			dataset2[user][movieID][0] *= normalized_value 


#年齢、性別、仕事でピアソンスコアに重み付け
def userfeature_adjust(otheruser, user_age, user_gender, user_occupation):
	for list in otheruser:
		user_number = list[1]
		otherage = userdata[user_number][0]
		if user_age == otherage:
			list[0] += fabs(list[0]*0.1)
		othergender = userdata[user_number][1]
		if user_gender == othergender:
			list[0] += fabs(list[0]*0.1)
		otheroccupation = userdata[user_number][2]
		if user_occupation == otheroccupation:
			list[0] += fabs(list[0]*0.1)

def then_evaluate():
	


"""
#映画固有の正規化評価/高い映画はいい映画だから高いから正規化する必要なし
def normalized_movie(bar):
	for i in range(1,bar+1):
		movie_ave = 0
		n = 0
		for user in dataset2:
			for movieID in dataset2[user]:
				if movieID == str(i):
					movie_ave += dataset2[user][movieID]
					n += 1
				else:
					continue
		normalized_value = (n/movie_ave)/glob_ave
		print normalized_value
		for user in dataset2:
			for movieID in dataset2[user]:
				if movieID == str(i):
					dataset2[user][movieID] *= normalized_value
				else:
					continue
"""

#No.196に協調フィルタ(評価値のみを利用)
def pearson_score(person):
	pearson_list = []
	for userID in range(1,944):
		other = str(userID)
		if other == person:
			continue

		#共通のアイテム
		both_view = {}

		#共通のアイテムあれば1なければ0を返す
		for item in dataset2[person]:
			if item in dataset2[other]:
				both_view[item] = 1

		if len(both_view) == 0:
			continue

		#和
		sumx = sum([dataset2[person][item][0] for item in both_view])
		sumy = sum([dataset2[other][item][0] for item in both_view])

		#積和
		sekiwa = sum([dataset2[person][item][0]*dataset2[other][item][0] for item in both_view])

		#和積
		waseki = (sumx*sumy)/len(both_view)

		#自乗和
		sqrt_sumx = sum([pow(dataset2[person][item][0], 2) for item in both_view])
		sqrt_sumy = sum([pow(dataset2[other][item][0], 2) for item in both_view])

		#Sxx,Syy
		Sxx = round(sqrt_sumx - (pow(sumx, 2)/len(both_view)), 2)
		Syy = round(sqrt_sumy - (pow(sumy, 2)/len(both_view)), 2)

		numerator = sekiwa - waseki

		denominator = sqrt(Sxx*Syy)
		if denominator == 0:
			continue
		else:
			r = numerator/denominator
		pearson_list.append([r, other])
	pearson_list.sort()
	pearson_list.reverse()
	#0~1に正規化 
	Max = max(pearson_list)[0]
	Min = min(pearson_list)[0]
	maxmin = Max-Min
	pearson_list = [[(x[0]-Min)/maxmin, x[1]] for x in pearson_list]
	return pearson_list

def recommened_coll(person, others):
	#各映画のピアソンスコア＊評価
	totals = {}
	simSUms = {}

	for i in others:
		if i[1] == person:
			continue

		for item in dataset2[i[1]]:
			if item not in dataset2[person] or dataset2[person][item][0] == 0:
				#点数がない、0のやつははじく
				if dataset2[i[1]][item][0] <= 0.0:
					continue
				if i[0] == 0.0:
					continue
				totals.setdefault(item, 0)
				totals[item] += dataset2[i[1]][item][0]*i[0]
				simSUms.setdefault(item, 0)
				simSUms[item] += i[0]

	#評価＊類似度を類似度の合計で割ることで、類似度で０〜５で正規化
	"""
	rankings = [[total/simSUms[item], item] for item, total in list(totals.items())]
	rankings.sort()
	rankings.reverse()
	print rankings[:5]
	"""

	#評価＊類似度を類似度の合計で割ることで、類似度で０〜５で正規化後、５で割り０〜１で正規化
	for item, total in list(totals.items()):
		if simSUms[item] == 0:
			print item
	#hybで指定できるようにitem先
	rankings = [[item, total/(simSUms[item])] for item, total in list(totals.items())]
	#推薦アイテム
	#recommendation_list = [recommend_item for score, recommend_item in rankings]
	return rankings

def contentbase_hyb(user, coll_list):
	movierating = 0
	movie = []
	coll_dict = dict(coll_list)

	for movieID in dataset2[user]:
		movie.append(movieID)
		#ユーザーのコンテンツにおいて、最近のほど評価高く
		rating = dataset2[user][movieID][0]*(1+(0.01*dataset2[user][movieID][1]))
		content = np.array(map(int,moviecontent[movieID]))
		movierating += rating*content

	averating = movierating/len(dataset2[user])

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
			if m in coll_dict:
				sim_list.append([sim*coll_dict[m], m])

	sim_list.sort()
	sim_list.reverse()
	return sim_list

#ユーザーとグローバルの差による正規化
normalized_user()

"""
#各映画とグローバルの差による正規化
normalized_movie(1682)
"""

#ピアソンスコア
pearson_list = pearson_score(str(196))

#pearson_scoreに職業、年齢、性別で重み付け
userfeature_adjust(pearson_list, userdata["196"][0], userdata["196"][1], userdata["196"][2])

#ピアソンスコアを基に協調フィルタで推薦
coll_list = recommened_coll(str(196), pearson_list)
#contentベース
content_hyb_list = contentbase_hyb(str(196), coll_list)

"""
print "No.196に合うTop5(coll)"
for score, number in recom_list[:5]:
	print score,moviename[number]



print "No.196におすすめTop5は(content)"
for sim, movieID in sim_list[:5]:
	print sim,moviename[movieID]
"""

#ハイブリッドベース：両方のコストを掛け合わせる
print "No.196におすすめTop5は(hyb)"
for sim, movieID in content_hyb_list[:5]:
	print sim,moviename[movieID]


