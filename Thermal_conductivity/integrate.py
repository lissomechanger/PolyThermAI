#!/usr/bin/env python3
import sys
import numpy as np
from scipy import integrate

def read_data(ifn, col):
    xlist = []
    ylist = []
    with open(ifn, 'r') as f:
        for line in f:
            words = line.split()
            if line.startswith('#'):
                continue
            xlist.append(float(words[0]))
            ylist.append(float(words[col]))
    return np.array(xlist), np.array(ylist)

dataset_x = []
dataset_y = []

corrfile=sys.argv[1]
for i in range(20):
    xlist, ylist = read_data(corrfile, i+1)
    dataset_x.append(xlist)
    dataset_y.append(ylist)

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)
n_set = dataset_x.shape[0]
N = dataset_x.shape[1]

vals = []
for i in range(n_set):
    y_int = integrate.cumtrapz(dataset_y[i], x=dataset_x[i], initial=0.0)
    vals.append(y_int)
vals = np.array(vals)
vals_mean = np.mean(vals, axis=0)
vals_final = vals_mean[int(0.9*N):]
vals_converge = np.mean(vals[:, int(0.9*N):], axis=1)

print('%12.3f %12.3f'%(np.average(vals_final), np.std(vals_converge)/np.sqrt(n_set-1)))

t = dataset_x[0]
with open ('hcf.txt', 'w') as f:
    for j in range(N):
        print(t[j], vals_mean[j], file=f)
    f.close()
