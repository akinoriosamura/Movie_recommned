# -*- coding:utf-8 -*-

#movie=[movieID,name,data,"",URL,content[5:]]
movie = []
for line in open("./ml-100k/u.item", "r"):
	itemlist = line[:-1].split("|")
	movie.append(itemlist)

moviename = {}
moviecontent = {}

for i in movie:
	moviename[i[0]] = i[1] 

for content in movie:
	moviecontent[content[0]]=content[5:]






