# -*- coding:utf-8 -*-

movie = []
userdata = {}
for line in open("./ml-100k/u.user", "r"):
	itemlist = line[:-1].split("|")
	userdata[itemlist[0]] = [itemlist[1], itemlist[2], itemlist[3]]
