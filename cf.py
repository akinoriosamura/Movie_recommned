# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from collections import defaultdict
import numpy

def jaccard(e1, e2):
	sete1 = set(e1)
	sete2 = set(e2)
	return float(len(sete1&sete2))/float(len(sete1|sete2))


product_x = [1, 3, 5]
product_a = [2, 4, 5]
product_b = [1, 2, 3]
product_c = [2, 3, 4, 7]
product_d = [3]
product_e = [4, 6, 7]

products = {
    'A': product_a,
    'B': product_b,
    'C': product_c,
    'D': product_d,
    'E': product_e,
}

print "共起"
r = defaultdict(int)

for key in products:
	overlap = list(set(product_x) & set(products[key]))
	print overlap
	r[key] = len(overlap)
print r

print "ジャッカード"
r2 = defaultdict(float)
for key in products:
	r2[key] = jaccard(product_x, products[key])
print r2