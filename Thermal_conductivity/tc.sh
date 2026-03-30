#!/bin/bash

corrfile=allcorr.data
kfile=kappa.txt

python calc_corr.py j_data ${corrfile} $1
python integrate.py ${corrfile} > ${kfile}
