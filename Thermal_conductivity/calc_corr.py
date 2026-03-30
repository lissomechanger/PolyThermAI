#!/usr/bin/env python3
import sys
import numpy as np
import subprocess
import MDAnalysis as mda

dest = sys.argv[1]
corrfile = open(sys.argv[2], "w")
out = subprocess.check_output(['ls', dest])
list_ifn = [ w.decode("utf-8") for w in out.split() ]

# Please revise the temperature, correlation length, time settings (Nc, Ns, dt) and other parameters according to your system and needs !!!
T = 298 # K
corr_length = 1000
Nc = 10
Ns = 1
dt = 0.5
kB = 1.3806504e-23 # SI, m2 kg s-2 K-1
A2m = 1.0e-10
fs2s = 1.0e-15
kCal2J = 4186.0/6.02214e23
u0 = mda.Universe(sys.argv[3])
a, b, c, alpha, beta, gamma = u0.dimensions
volume = a*b*c

convert = kCal2J*kCal2J/fs2s/A2m
scale = convert/(kB*T*T*volume)

def autocorr(x):
    N = len(x)
    n = corr_length
    result = np.correlate(x, x, mode='full')
    tcf = result[result.size // 2:(result.size//2+n)]
    tcf /= np.linspace(N, N-n+1, 1)
    return tcf

def read_J_traj(ifn):
    t = [] # in ps
    J = []
    with open(ifn, 'r') as f:
        for line in f:
            words = line.split()
            t.append(float(words[0]))
            J.append([float(w) for w in words[1:4]])
    t = np.array(t)
    J = np.array(J)
    return t, J

corrs = []
idata = 0
for ifn in list_ifn:
    # print auto correlation, normalize
    ts, Js = read_J_traj(dest+'/'+ifn)
    corrx = autocorr(Js[:, 0])
    corry = autocorr(Js[:, 1])
    corrz = autocorr(Js[:, 2])
    corr = (corrx + corry + corrz) / 3
    corrs.append(corr)

corrs = np.array(corrs)*scale
corr = np.average(corrs, axis=0)

ts = np.arange(0, corr_length)*Ns*Nc*dt  # total 3ps
for i in range(corr_length):
    t = ts[i]
    print(t, *corrs[:, i], file=corrfile)

with open ('corr.txt', 'w') as f:
    for i in range(corr_length):
        t = ts[i]
        c = corr[i]
        print(t, c, file=f)
    f.close()

