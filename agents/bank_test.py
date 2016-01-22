#!/usr/bin/python

import numpy as np
from transition_bank import TransitionBank
from time import time

batches = 2000
batch_size = 40
insertions = 10000
capacity = insertions


format = dict()
format["s_img"] = (3,45,60)
format["s_misc"] = 3


bank = TransitionBank(format, capacity, batch_size)

s1 = np.float32(np.random.random(format["s_img"]))
s2 = np.float32(np.random.random(format["s_img"]))
misc1 = np.float32(np.random.rand(format["s_misc"]))
misc2 = np.float32(np.random.rand(format["s_misc"]))
print misc1
r = np.random.rand(1)
a = np.random.rand(1)

print "Capacity:", capacity

start = time()
if format["s_misc"]>0:
    for i in range(insertions):
        bank.add_transition([s1,misc1], a, [s2,misc2], r)
else:
    for i in range(insertions):
        bank.add_transition([s1], a, [s2], r)
end = time()
print "Insertions:", insertions, "time: ", round(end - start, 2), "s"



if batch_size>0 and batches>0:
    start = time()
    for i in range(batches):
        sample = bank.get_sample()
    end = time()
    print "batches:",batches,"batchsize:", batch_size, "time: ", round(end - start, 2), "s"

