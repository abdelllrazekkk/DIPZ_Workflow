#Utilities to be used in the mlpl functions
from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import time

# A function that takes in the event number and calculates the maximum over 4-jet combinations of log of the product of the likelihood functions of all the jets in the event ANALYTICALLY
def get_max_log_likelihood_an(event_id,jets,comb_num,num_highest_pt=555555):
    event_jets = jets[jets["eventNumber"] == event_id]
    
    if num_highest_pt != 555555:
        if len(event_jets) > num_highest_pt:
            event_jets = event_jets[(-event_jets['pt']).argsort()[:num_highest_pt]]
    
    combinations = []
    mlpl_array = []

    for combination in it.combinations(event_jets, comb_num):
        combinations.append(combination)

    for comb in combinations:
        num = 0
        denom = 0
        second_term = 0
        third_term = 0
        for jet in comb:
            mu = jet["dipz20230223_z"] * 50
            sigma = np.exp(-0.5*jet["dipz20230223_negLogSigma2"]) * 50

            num += (mu) / (sigma**2)
            denom += 1 / (sigma**2)
            second_term -= np.log(sigma)
        for jet in comb:
            mu = jet["dipz20230223_z"] * 50
            sigma = np.exp(-0.5*jet["dipz20230223_negLogSigma2"]) * 50

            third_term -= ((num / denom) - mu)**2 / (2*sigma**2) 
            
        mlpl_array.append(-4 * np.log(np.sqrt(2*np.pi)) + second_term + third_term)
            
    
    max_log_likelihood = max(mlpl_array)
        
    return max_log_likelihood


def get_max_log_likelihood_an_hh4b(event_id,jets,comb_num,num_highest_pt=555555):
    event_jets = jets[jets["eventNumber"] == event_id]

    if num_highest_pt != 555555:
        if len(event_jets) > num_highest_pt:
            event_jets = event_jets[(-event_jets['pt']).argsort()[:num_highest_pt]]
    
    combinations = []

    for combination in it.combinations(event_jets, comb_num):
        combinations.append(combination)

    mlpl = -999999999999
    num_bjets_chosen = 0

    for comb in combinations:
        num = 0
        denom = 0
        second_term = 0
        third_term = 0
        num_bjets = 0 
        for jet in comb:
            mu = jet["dipz20230223_z"] * 50
            sigma = np.exp(-0.5*jet["dipz20230223_negLogSigma2"]) * 50

            num += (mu) / (sigma**2)
            denom += 1 / (sigma**2)
            second_term -= np.log(sigma)
            
        for jet in comb:
            mu = jet["dipz20230223_z"] * 50
            sigma = np.exp(-0.5*jet["dipz20230223_negLogSigma2"]) * 50

            third_term -= ((num / denom) - mu)**2 / (2*sigma**2)

            if jet['HadronConeExclTruthLabelID'] == 5:
               num_bjets += 1
    
        if -4 * np.log(np.sqrt(2*np.pi)) + second_term + third_term > mlpl:
            mlpl = -4 * np.log(np.sqrt(2*np.pi)) + second_term + third_term
            num_bjets_chosen = num_bjets
    
    max_log_likelihood = mlpl
    
    return max_log_likelihood, num_bjets_chosen

# A function that takes in a h5 sample name and outputs a list of the discriminant variable (MLPL) values over all the events in the sample
def get_max_log_likelihood_dist(name, comb_num, num=999999999999999 ,num_highest_pt=555555):
    start = time.time()
    data = File(name, 'r')
    jets = data['jets']
    uniques = np.unique(jets["eventNumber"])
    
    print("The number of jets in the sample is: " + str(len(jets)))
    print("The number of jets in the sample with pT < 20 GeV is: " + str(len(jets[jets["pt"] < 20])))
    print("The number of jets in the sample with eta > 2.5 GeV is: " + str(len(jets[jets["eta"] > 2.5])))
    print("The number of events in our sample is: " + str(len(uniques)))
    
    max_log_likelihood_list = []
    jet_multiplicities = []
    no_of_processed_events = num
    counter = 0

    for id in uniques:
        jet_multiplicities.append(len(jets[jets["eventNumber"] == id]))
        if len(jets[jets["eventNumber"] == id]) >= 4:
            max_log_likelihood_list.append(get_max_log_likelihood_an(id,jets,comb_num,num_highest_pt))
            counter +=1
        if counter == no_of_processed_events:
            break

    if counter == no_of_processed_events:
        print("The provided number of four or more jet events in the sample was run over and it is: " + str(num))
    if counter != no_of_processed_events:
        print("The number of four or more jet events in the sample is: " + str(counter))
        print("The number of four or more jet events in the sample is less than the provided number, therefore all the sample was run over.")
    end = time.time()
    print("The time of execution of the (get_max_log_likelihood_dist) function for the (" + name + ") file is :", ((end-start) / 60) , "min")

    return max_log_likelihood_list, jet_multiplicities

def get_max_log_likelihood_dist_hh4b(name, comb_num, num=999999999999999, num_highest_pt=555555):
    start = time.time()
    data = File(name, 'r')
    jets = data['jets']
    uniques = np.unique(jets["eventNumber"])
    
    print("The number of jets in the sample is: " + str(len(jets)))
    print("The number of jets in the sample with pT < 20 GeV is: " + str(len(jets[jets["pt"] < 20])))
    print("The number of jets in the sample with eta > 2.5 GeV is: " + str(len(jets[jets["eta"] > 2.5])))
    print("The number of events in our sample is: " + str(len(uniques)))
    
    max_log_likelihood_list = []
    num_bjets_chosen_list = []
    jet_multiplicities = []
    no_of_processed_events = num
    counter = 0

    for id in uniques:
        jet_multiplicities.append(len(jets[jets["eventNumber"] == id]))
        event_jets = jets[jets["eventNumber"] == id]
        bjets = event_jets[event_jets["HadronConeExclTruthLabelID"] == 5]
        if len(event_jets) >= 4 & len(bjets) >= 4:
            max_log_likelihood, num_bjets_chosen = get_max_log_likelihood_an_hh4b(id,jets,comb_num,num_highest_pt)
            max_log_likelihood_list.append(max_log_likelihood)
            num_bjets_chosen_list.append(num_bjets_chosen)
            counter +=1
        if counter == no_of_processed_events:
            break

    if counter == no_of_processed_events:
        print("The provided number of four or more b-jet events in the sample was run over and it is: " + str(num))
    if counter != no_of_processed_events:
        print("The number of four or more b-jet events in the sample is: " + str(counter))
        print("The number of four or more b-jet events in the sample is less than the provided number, therefore all the sample was run over.")
    end = time.time()
    print("The time of execution of the (get_max_log_likelihood_dist) function is :", ((end-start) / 60) , "min")

    return max_log_likelihood_list, jet_multiplicities, num_bjets_chosen_list