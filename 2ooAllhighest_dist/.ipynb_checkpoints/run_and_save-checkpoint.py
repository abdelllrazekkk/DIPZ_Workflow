#!/usr/bin/env python
# coding: utf-8

"""Importing all needed packages"""
from h5py import File
import numpy as np
import itertools as it
import time
import joblib
from a import get_max_log_likelihood_an
from a import get_max_log_likelihood_an_hh4b
from a import get_max_log_likelihood_dist
from a import get_max_log_likelihood_dist_hh4b


"""Setting up needed parameters"""
comb_num = 2
max_num = 9999999
num_highest_pt = 555555


EB_dist = get_max_log_likelihood_dist("../../EBdata.h5", comb_num, max_num, num_highest_pt)
joblib.dump(EB_dist, 'EB_dist.sav')

hh4b_dist, num_bjets_chosen_list= get_max_log_likelihood_dist_hh4b("../../hh4b.h5", comb_num, max_num, num_highest_pt)
joblib.dump(hh4b_dist, 'hh4b_dist_nofake.sav')
joblib.dump(num_bjets_chosen_list, 'num_bjets_chosen_list.sav')

jz0_dist = get_max_log_likelihood_dist("../../jz0.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz0_dist, 'jz0_dist.sav')

jz1_dist = get_max_log_likelihood_dist("../../jz1.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz1_dist, 'jz1_dist.sav')

jz2_dist = get_max_log_likelihood_dist("../../jz2.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz2_dist, 'jz2_dist.sav')

jz3_dist = get_max_log_likelihood_dist("../../jz3.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz3_dist, 'jz3_dist.sav')

jz4_dist = get_max_log_likelihood_dist("../../jz4.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz4_dist, 'jz4_dist.sav')

jz5_dist = get_max_log_likelihood_dist("../../jz5.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz5_dist, 'jz5_dist.sav')

