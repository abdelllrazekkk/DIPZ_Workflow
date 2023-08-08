#!/usr/bin/env python
# coding: utf-8
from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import random
import itertools as it
import time
import joblib

start = time.time()

def get_dists(string, max_num=60000):
    data = File(string, 'r')
    jets = data['jets']
    jets = np.asarray(jets)
    uniques = np.unique(jets["eventNumber"])
        
    counter = 0
    jet_multiplicities = []
    leading_pT = []
    subleading_pT = []

    for i in uniques:
        jet_multiplicities.append(len(jets[jets["eventNumber"] == i]))
        pTs = []
        event_jets = jets[jets["eventNumber"] == i]
        for jet in event_jets:
            pTs.append(jet["pt"])
        leading_pT.append(max(pTs))
        if len(pTs) > 1: 
            subleading_pT.append(sorted(pTs)[-2])
        counter += 1
        if counter == max_num:
            break 
    return jet_multiplicities, leading_pT, subleading_pT


hh4b_multiplicities, hh4b_leading_pt_dist, hh4b_subleading_pt_dist = get_dists('../hh4b.h5')
EBdata_multiplicities, EBdata_leading_pt_dist, EBdata_subleading_pt_dist = get_dists('../EBdata.h5')
jz0_multiplicities, jz0_leading_pt_dist, jz0_subleading_pt_dist = get_dists('../jz0.h5')
jz1_multiplicities, jz1_leading_pt_dist, jz1_subleading_pt_dist = get_dists('../jz1.h5')
jz2_multiplicities, jz2_leading_pt_dist, jz2_subleading_pt_dist = get_dists('../jz2.h5')
jz3_multiplicities, jz3_leading_pt_dist, jz3_subleading_pt_dist = get_dists('../jz3.h5')
jz4_multiplicities, jz4_leading_pt_dist, jz4_subleading_pt_dist = get_dists('../jz4.h5')
jz5_multiplicities, jz5_leading_pt_dist, jz5_subleading_pt_dist = get_dists('../jz5.h5')

joblib.dump(hh4b_multiplicities, './dist_jet_multiplicities/hh4b_multiplicities.sav')
joblib.dump(EBdata_multiplicities, './dist_jet_multiplicities/EBdata_multiplicities.sav')
joblib.dump(jz0_multiplicities, './dist_jet_multiplicities/jz0_multiplicities.sav')
joblib.dump(jz1_multiplicities, './dist_jet_multiplicities/jz1_multiplicities.sav')
joblib.dump(jz2_multiplicities, './dist_jet_multiplicities/jz2_multiplicities.sav')
joblib.dump(jz3_multiplicities, './dist_jet_multiplicities/jz3_multiplicities.sav')
joblib.dump(jz4_multiplicities, './dist_jet_multiplicities/jz4_multiplicities.sav')
joblib.dump(jz5_multiplicities, './dist_jet_multiplicities/jz5_multiplicities.sav')

joblib.dump(hh4b_leading_pt_dist, './dist_leading_pt/hh4b_leading_pt_dist.sav')
joblib.dump(EBdata_leading_pt_dist, './dist_leading_pt/EBdata_leading_pt_dist.sav')
joblib.dump(jz0_leading_pt_dist, './dist_leading_pt/jz0_leading_pt_dist.sav')
joblib.dump(jz1_leading_pt_dist, './dist_leading_pt/jz1_leading_pt_dist.sav')
joblib.dump(jz2_leading_pt_dist, './dist_leading_pt/jz2_leading_pt_dist.sav')
joblib.dump(jz3_leading_pt_dist, './dist_leading_pt/jz3_leading_pt_dist.sav')
joblib.dump(jz4_leading_pt_dist, './dist_leading_pt/jz4_leading_pt_dist.sav')
joblib.dump(jz5_leading_pt_dist, './dist_leading_pt/jz5_leading_pt_dist.sav')


joblib.dump(hh4b_subleading_pt_dist, './dist_subleading_pt/hh4b_subleading_pt_dist.sav')
joblib.dump(EBdata_subleading_pt_dist, './dist_subleading_pt/EBdata_subleading_pt_dist.sav')
joblib.dump(jz0_subleading_pt_dist, './dist_subleading_pt/jz0_subleading_pt_dist.sav')
joblib.dump(jz1_subleading_pt_dist, './dist_subleading_pt/jz1_subleading_pt_dist.sav')
joblib.dump(jz2_subleading_pt_dist, './dist_subleading_pt/jz2_subleading_pt_dist.sav')
joblib.dump(jz3_subleading_pt_dist, './dist_subleading_pt/jz3_subleading_pt_dist.sav')
joblib.dump(jz4_subleading_pt_dist, './dist_subleading_pt/jz4_subleading_pt_dist.sav')
joblib.dump(jz5_subleading_pt_dist, './dist_subleading_pt/jz5_subleading_pt_dist.sav')

finish = time.time()

print("The time of execution of the (get_dists.py) is:", round(finish-start / 60, 2) , "min")