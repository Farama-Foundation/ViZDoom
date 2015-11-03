#!/usr/bin/python

import numpy as np
from transition_bank import TransitionBank
from time import time
from time import sleep

insertions = 10000
capacity = insertions 
batch_size = 10000

if insertions > capacity:
	overflowed_insertions = insertions - capacity
	insertions -= overflowed_insertions
else:
	overflowed_insertions = 0

bank = TransitionBank(capacity)

s1 = np.float32(np.random.rand(4,120,90))
s2 = np.float32(np.random.rand(4,120,90))
r = np.float32(1)
a = np.float32(2)

print "Capacity:",capacity

start = time()
for i in range(insertions):
	bank.add_transition(s1.copy(),a,s2.copy(),r)

end = time()
print "Insertions:",insertions, "time: ",round(end-start,2),"s"

if overflowed_insertions > 0:
	start = time()
	for i in range(overflowed_insertions):
		bank.add_transition(s1.copy(),a,s2.copy(),r)
	end = time()
	print "Overflowed insertions:",overflowed_insertions, "time: ",round(end-start,2),"s"


if batch_size >0:
	start = time()
	batch = bank.get_sample(batch_size)
	end = time()
	print "Batchsize:",batch_size, "time: ",round(end-start,2),"s"
