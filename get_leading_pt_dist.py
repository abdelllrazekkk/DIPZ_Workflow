#!/usr/bin/env python
# coding: utf-8

# # Tests of the event rejection (ER) algorithm for four-jet event selections

# In[30]:


from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import random
import itertools as it
import time
import joblib


def get_leading_pt_dist(string, max_num=60000):
    data = File(string, 'r')
    jets = data['jets']
    uniques = np.unique(jets["eventNumber"])
        
    counter = 0
    leading_pT = []
    
    for i in uniques:
        pTs = []
        event_jets = jets[jets["eventNumber"] == i]
        for jet in event_jets:
            pTs.append(jet["pt"])
        leading_pT.append(max(pTs))
        counter += 1
        if counter == max_num:
            break 
    return leading_pT


# In[25]:


leading_pT_jz0_dist = get_leading_pt_dist('../jz0.h5')
joblib.dump(leading_pT_jz0_dist, 'leading_pT_jz0_dist.sav')


# In[ ]:


leading_pT_jz1_dist = get_leading_pt_dist('../jz1.h5')
joblib.dump(leading_pT_jz1_dist, 'leading_pT_jz1_dist.sav')


# In[ ]:


leading_pT_jz2_dist = get_leading_pt_dist('../jz2.h5')
joblib.dump(leading_pT_jz2_dist, 'leading_pT_jz2_dist.sav')


# In[ ]:


leading_pT_jz3_dist = get_leading_pt_dist('../jz3.h5')
joblib.dump(leading_pT_jz3_dist, 'leading_pT_jz3_dist.sav')


# In[ ]:


leading_pT_jz4_dist = get_leading_pt_dist('../jz4.h5')
joblib.dump(leading_pT_jz4_dist, 'leading_pT_jz4_dist.sav')


# In[ ]:


leading_pT_jz5_dist = get_leading_pt_dist('../jz5.h5')
joblib.dump(leading_pT_jz5_dist, 'leading_pT_jz5_dist.sav')
