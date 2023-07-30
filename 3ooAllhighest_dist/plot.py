import matplotlib.pyplot as plt
from h5py import File


#A function that takes in a distribution and plots it
def plot_distribution(max_log_likelihood_list):
    plt.hist(max_log_likelihood_list, bins = 500)
    plt.yscale("log")
    plt.xlabel("MLPL", loc='right')
    plt.title('Distribution of the discriminant variable (MLPL)')
    plt.show()


#A function that outputs an array of the jet multiplicities in the sample
def get_jetmultiplicities(name):
    data = File(name, 'r')
    jets = data['jets']
    uniques = np.unique(jets["eventNumber"])
    jet_multiplicities = []
    
    for i in uniques:
        jet_multiplicities.append(len(jets[jets["eventNumber"] == i]))

    return jet_multiplicities