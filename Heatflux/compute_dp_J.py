#!/usr/bin/env python3

import os, sys
import numpy as np
from deepmd.infer import DeepPot as DP
import time

m_per_s_2_A_per_fs = 0.00001
eV2kCal = 23.06031
g_A2_per_fs2_to_kcal = 2390.1
Nc = 1

def read(fname):
    ifile = open(fname, "r")
    a_type = []

    iline = ifile.readlines() # I hate wide chars (doubled memory cost)
    Natom = int(iline[0])
    #CELL = np.diag(np.array(list(map(float,iline[1].split()[2:5]))))

    nframe = len(iline)//(Natom+2)
    data = np.zeros([nframe,Natom,3],'float32')
    iStarT = np.array(range(2,len(iline),Natom+2)) # StarTline of each xyz frame content
    #iStarT = iStarT[:-1] # remove last element
    if len(iStarT) != nframe:
        print("something happened")
        exit()

    for iframe in range(0,len(iStarT)):
        ist = iStarT[iframe]
        for iatom in range(0, Natom):
            words = iline[ist+iatom].split()
            data[iframe][iatom] = list(map(float,words[1:4]))

    # get atype
    for i in range(iStarT[0],iStarT[0]+Natom):
        words = iline[i].split()
        if words[0] == 'O' or words[0] == 'OW':
            a_type.append(2)
        elif words[0] == 'H' or words[0] == 'D' or words[0] == 'DW':
            a_type.append(0)
        elif words[0] == 'C':
            a_type.append(1)
        else:
            print("unknown atom type")
            exit()

    ifile.close()
    return np.array(data,'float32'), np.array(a_type)

def dp_eval(pos):
    e, f, v, ae, av = dp.eval(pos[None,:], box[None,:], atype, atomic = True)
    return ae[0] * eV2kCal, av[0].reshape([Natom, 3,3]) * eV2kCal

def cal_J_sub(i_ae, i_av, vel, index):
    ae_j = i_ae[index]
    av_j = i_av[index]
    vel_j = vel[index]
    #ae_j = [i_ae[i] for i in index]
    #av_j = [i_av[i, :, :] for i in index]
    #vel_j = np.array([vel[i, :] for i in index])
    J_convect = np.dot(ae_j.T, vel_j)
    J_conduct = np.dot(vel_j.reshape(1, -1), av_j.reshape(-1, 3))
    J = J_convect + J_conduct
    return J[0]

def cal_mass(atype):
    mass = np.array(atype)
    mass[np.where(atype==0)[0]]=1.00792 # H
    mass[np.where(atype==1)[0]]=12.0107 # C
    mass[np.where(atype==2)[0]]=15.9994 # O
    return mass


if __name__ == '__main__':

    i = int(sys.argv[1])
    rootpath = "./"
    xfile = "pos"
    vfile = "vel"
    start=time.time()

    positions = np.load(rootpath+xfile+".npy")
    boxes = np.load(rootpath+"box.npy")
    velocities = np.load(rootpath+vfile+".npy")
    atype = np.load(rootpath+"atype.npy")
    box=boxes[0]
    Nframes = positions.shape[0]
    Natom = positions.shape[1]
    velocities = velocities * m_per_s_2_A_per_fs  ## unit is angstrom/fs
    dp = DP(rootpath+'graph.pb')

    # atype: (natoms,)
    H_index = np.where(atype==0)[0]
    C_index = np.where(atype==1)[0]
    O_index = np.where(atype==2)[0]

    # mass: (natoms,)
    mass = cal_mass(atype)

    # ae_list: nframe*natom
    ae_list = np.zeros([Nframes,Natom],'float32')
    # av_list: nframe*natom*3*3
    av_list = np.zeros([Nframes,Natom,3,3],'float32')

    start=time.time()
    for iframe in range(0, Nframes, Nc):
        vel = velocities[iframe]
        i_ape, i_av = dp_eval(positions[iframe]) # kcal/mol
        # i_ape: natom*1; i_av: natom*3*3; vel: natom*3
        i_ake = g_A2_per_fs2_to_kcal*mass*(np.linalg.norm(vel,None,1)**2)/2
        ae_list[iframe] = np.array(i_ape).T+i_ake # 1*natom
        av_list[iframe] = i_av # natom*3*3

    # remove energy bias
    ae_list = ae_list - ae_list.mean(0)

    start=time.time()
    for iframe in range(0, Nframes, Nc):
        vel = velocities[iframe]
        J_convect = np.dot(ae_list[iframe], vel)
        J_conduct = np.dot(vel.reshape(1, -1), av_list[iframe].reshape(-1, 3))
        J = J_convect + J_conduct
        print(iframe, J[0,0], J[0,1], J[0,2])
