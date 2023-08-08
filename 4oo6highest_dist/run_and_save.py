#!/usr/bin/env python
# coding: utf-8

# # Getting discriminant variable (MLPL) distributions and jet multiplicities

# In[1]:


"""Importing all needed packages"""
from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import random
import itertools as it
import time
import joblib
from plot import plot_distribution
from plot import get_jetmultiplicities
from a import get_max_log_likelihood_an
from a import get_max_log_likelihood_an_hh4b
from a import get_max_log_likelihood_dist
from a import get_max_log_likelihood_dist_hh4b


# In[2]:


"""Setting up needed parameters"""
comb_num = 4
max_num = 9999999
num_highest_pt = 6


# In[4]:


EB_dist, EB_multiplicities = get_max_log_likelihood_dist("../../EBdata.h5", comb_num, max_num, num_highest_pt)
joblib.dump(EB_dist, 'EB_dist.sav')
joblib.dump(EB_multiplicities, 'EB_multiplicities.sav')
hh4b_dist, hh4b_multiplicities, num_bjets_chosen_list= get_max_log_likelihood_dist_hh4b("../../hh4b.h5", comb_num, max_num, num_highest_pt)
joblib.dump(hh4b_dist, 'hh4b_dist_nofake.sav')
joblib.dump(hh4b_multiplicities, 'hh4b_multiplicities.sav')
joblib.dump(num_bjets_chosen_list, 'num_bjets_chosen_list.sav')
jz0_dist, jz0_multiplicities = get_max_log_likelihood_dist("../../jz0.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz0_dist, 'jz0_dist.sav')
joblib.dump(jz0_multiplicities, 'jz0_multiplicities.sav')
jz1_dist, jz1_multiplicities = get_max_log_likelihood_dist("../../jz1.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz1_dist, 'jz1_dist.sav')
joblib.dump(jz1_multiplicities, 'jz1_multiplicities.sav')
jz2_dist, jz2_multiplicities = get_max_log_likelihood_dist("../../jz2.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz2_dist, 'jz2_dist.sav')
joblib.dump(jz2_multiplicities, 'jz2_multiplicities.sav')
jz3_dist, jz3_multiplicities = get_max_log_likelihood_dist("../../jz3.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz3_dist, 'jz3_dist.sav')
joblib.dump(jz3_multiplicities, 'jz3_multiplicities.sav')
jz4_dist, jz4_multiplicities = get_max_log_likelihood_dist("../../jz4.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz4_dist, 'jz4_dist.sav')
joblib.dump(jz4_multiplicities, 'jz4_multiplicities.sav')
jz5_dist, jz5_multiplicities = get_max_log_likelihood_dist("../../jz5.h5", comb_num, max_num, num_highest_pt)
joblib.dump(jz5_dist, 'jz5_dist.sav')
joblib.dump(jz5_multiplicities, 'jz5_multiplicities.sav')

