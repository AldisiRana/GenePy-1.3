# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:51:25 2019
@author: Enrico
"""

freq = ([line.strip('chr').rstrip().split("\t") for line in open("freqanno", "r")])
cadd = ([line.rstrip().split("\t") for line in open("caddout.tsv", "r")])

cadd.pop(0)
# cadd.pop(0)
# freq.pop(0)

F = []
C = []

for i in freq[1:]:
    F.append("-".join(i[:2]))
for i in cadd[1:]:
    C.append(["-".join(i[:2]), i[-2]])

dC = dict(C)
# %%

O = []
i = 0
for k in F:
    try:
        O.append(dC[k])
    except:
        i += 1
        O.append("NAN")

import numpy as np

O = np.array(O).astype(np.object)

F = np.array(F)

for i in np.where(O == "NAN")[0]:
    try:
        x = F[i]
        x = x.split("-")
        x[-1] = str(int(x[-1]) - 1)
        x = "-".join(x)
        O[i] = dC[x]
    except:
        print('Fail to annotate %s' % (F[i]))

out = O[:]
out = np.insert(out, 0, "RawScore")
np.savetxt("caddanno", out, fmt='%s')