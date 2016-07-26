# -*- coding: utf-8 -*-    

from recomdata import dataset
from math import sqrt

def similarity_score(person1, person2):
	#戻り値はユークリッド距離

	#共通のアイテム辞書
	both_viewed = {}

	#共通のアイテムあれば1
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1

	#共通のなかったら0をreturn
	if len(both_viewed) == 0:
		return 0

	#ユークリッド距離
	eclidean = []

	#1つずつ計算し追加していく
	for key,value in both_viewed.iteritems():
		eclidean.append(pow(dataset[person1][key] - dataset[person2][key], 2))
	total = sum(eclidean)

	#ただのルートを返すと0の時が一番値が良くなる。先に0のロキは返しこちらで値が大きいほど良いように変換することで解決
	return 1/(1+sqrt(total))

def pearson_correlation(person1, person2):
	#共通のアイテム
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1

	#なければ0
	if len(both_rated) == 0:
		return 0
	
	#ユーザーの好みの合計
	person1_preference_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preference_sum = sum([dataset[person2][item] for item in both_rated])

	#xyの積和
	person12_sqrt_preference_sum = sum([dataset[person1][item]*dataset[person2][item] for item in both_rated])

	#xyのわ積
	person12 = (person1_preference_sum*person2_preference_sum)/len(both_rated)

	#2乗和
	person1_sqrt_preference_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
	person2_sqrt_preference_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])
	
	#Sxx,Sxy
	Sxx = person1_sqrt_preference_sum - pow(person1_preference_sum,2)/len(both_rated)
	Syy = person2_sqrt_preference_sum - pow(person2_preference_sum,2)/len(both_rated)

	#ピアソンスコアの計算
	#分子の計算
	numerator = person12_sqrt_preference_sum - person12

	#分母の計算
	denominator = sqrt(Sxx*Syy)
	#ピアソンスコア
	if denominator ==0:
		return 0
	else:
		r = numerator/denominator
		return r

def most_similar_user(person, number_of_user):
	scores = [(pearson_correlation(person, other_person), other_person) for other_person in dataset if other_person != person]

	scores.sort()
	scores.reverse()
	return scores[0:number_of_user]

def recommendation(person):
	totals = {}
	simSums = {}

	for other in dataset:
		if other == person:
			continue
		sim = pearson_correlation(person, other)
		print sim

		if sim <= 0:
			continue

		for item in dataset[other]:

			if item not in dataset[person] or dataset[person][item] == 0:
				totals.setdefault(item, 0)
				#他の人の評価✕他の人との類似度
				totals[item] += dataset[other][item]*sim
				#類似度の和
				simSums.setdefault(item, 0)
				simSums[item] += sim

	#ランキングリスト
	rankings = [(total/simSums[item], item) for item, total in list(totals.items())]
	rankings.sort()
	rankings.reverse()

	#推薦アイテム
	recommendations_list = [recommend_item for score, recommend_item in rankings]
	return recommendations_list


print "山田さんと鈴木さんの類似度(ユークリッド距離)",
print similarity_score("山田", "鈴木")

print "山田さんと田中さんの類似度 (ピアソン相関係数)",
print pearson_correlation('山田', '田中')

print "山田さんに似た人ベスト 3"
print most_similar_user("山田", 3)

print "下林さんにおすすめのメニュー",
print recommendation('下林')