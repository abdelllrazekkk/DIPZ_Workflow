#!/usr/bin/env python3
from turtle import title
from h5py import File
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#Opening the HDF5 File
input_file = "user.viruelas.27383479._000001.output.h5"
data = File(input_file, 'r')

#Accessing the 'Jets' Dataset
print("List of available datasets: " + str(list(data.keys())))
jets = data['jets']

#Printing the 'jets' Attributes
print("The shape of the dataset (jets): " + str(jets.shape))
print("The size of the dataset (jets): " + str(jets.size))
print("The number of dimensions of the dataset (jets): " + str(jets.ndim))
print("The attributes of each element of the dataset (jets): " + str(jets.dtype.fields.keys()))
print("The total number of bytes required to load the full dataset into RAM of the dataset (jets): " + str(jets.nbytes))



# Same for different Tracks datasets
tracks = data['tracks']
#print("The shape of the dataset (tracks): " + str(tracks.shape))
#print("The attributes of each element of the dataset (tracks): " + str(tracks.dtype.fields.keys()))
fs_tracks = data['fs_tracks']
#print("The shape of the dataset (fs_tracks): " + str(fs_tracks.shape))
#print("The attributes of each element of the dataset (fs_tracks): " + str(fs_tracks.dtype.fields.keys()))
fs_tracks_simple_ip = data['fs_tracks_simple_ip']
#print("The shape of the dataset (fs_tracks_simple_ip): " + str(fs_tracks_simple_ip.shape))
#print("The attributes of each element of the dataset (fs_tracks_simple_ip): " + str(fs_tracks_simple_ip.dtype.fields.keys()))

#Printing the track information for just the first jet in the dataset
'''
first = jets[0]     #First index is ZERO
print("The number of tracks of the first jet in the (jets) dataset is: " + str(first['n_tracks']))
print("The number of fs_tracks of the first jet in the (jets) dataset is: " + str(first['n_fs_tracks']))
print("The number of fs_tracks_simple_ip of the first jet in the (jets) dataset is: " + str(first['n_fs_tracks_simple_ip']))
'''

#Making histograms for the number of tracks, fs_tracks, and fs_tracks_simple_ip of all the jets. 
'''
plt.hist([jets['n_tracks'],jets['n_fs_tracks'],jets['n_fs_tracks_simple_ip']], range=[0,40], label=['Tracks','fs_tracks','fs_simple_ip'])
plt.yscale("log")
plt.legend(loc='upper right')
plt.title('Histogram for the Number of tracks Per jet for different types of tracks')
plt.savefig('Number of Tracks Per Jet.png')
plt.cla()
'''
#Making a histogram of, e.g., the track pt & track eta for the tracks in the first jet.
'''
first_jet_valid_pts = tracks[0][tracks[0]['valid']]['pt']
plt.hist(first_jet_valid_pts/1000.)
plt.title('Histogram for the pT of the First Jet Tracks')
plt.savefig('FirstJetTracks_pt.png')
plt.cla()
first_jet_valid_etas = tracks[0][tracks[0]['valid']]['eta']
plt.hist(first_jet_valid_etas, bins=10, range=[-5,5])
plt.title('Histogram for the eta of the First Jet Tracks')
plt.savefig('FirstJetTracks_eta.png')
plt.cla()
'''

#Making histograms for all the tracks variables
'''
tracks_hist_names = ['chiSquared', 'numberDoF', 'IP3D_signed_d0', 'IP2D_signed_d0', 'IP3D_signed_z0', 'phi', 'theta', 'qOverP', 'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits', 'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits', 'numberOfPixelHits', 'numberOfPixelHoles', 'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTHits', 'numberOfSCTHoles', 'numberOfSCTSharedHits', 'd0', 'z0SinTheta', 'IP3D_signed_d0_significance', 'IP3D_signed_z0_significance', 'pt', 'eta', 'd0Uncertainty', 'z0SinThetaUncertainty', 'z0RelativeToBeamspot', 'z0RelativeToBeamspotUncertainty', 'deta', 'dphi', 'dr', 'ptfrac', 'valid']
fs_tracks_hist_names = ['chiSquared', 'numberDoF', 'IP3D_signed_d0', 'IP2D_signed_d0', 'IP3D_signed_z0', 'phi', 'theta', 'qOverP', 'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits', 'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits', 'numberOfPixelHits', 'numberOfPixelHoles', 'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTHits', 'numberOfSCTHoles', 'numberOfSCTSharedHits', 'd0', 'z0SinTheta', 'IP3D_signed_d0_significance', 'IP3D_signed_z0_significance', 'pt', 'eta', 'd0Uncertainty', 'z0SinThetaUncertainty', 'z0RelativeToBeamspot', 'z0RelativeToBeamspotUncertainty', 'deta', 'dphi', 'dr', 'ptfrac', 'valid']
fs_tracks_simple_ip_hist_names = ['chiSquared', 'numberDoF', 'IP3D_signed_d0', 'IP2D_signed_d0', 'IP3D_signed_z0', 'phi', 'theta', 'qOverP', 'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits', 'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits', 'numberOfPixelHits', 'numberOfPixelHoles', 'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTHits', 'numberOfSCTHoles', 'numberOfSCTSharedHits', 'd0', 'z0SinTheta', 'IP3D_signed_d0_significance', 'IP3D_signed_z0_significance', 'pt', 'eta', 'd0Uncertainty', 'z0SinThetaUncertainty', 'z0RelativeToBeamspot', 'z0RelativeToBeamspotUncertainty', 'deta', 'dphi', 'dr', 'ptfrac', 'valid']

for name in tracks_hist_names: 
    var_tracks = tracks[name].flatten()
    var_fs_tracks = fs_tracks[name].flatten()
    var_fs_tracks_simple_ip = fs_tracks_simple_ip[name].flatten()
    plt.hist([var_tracks,var_fs_tracks,var_fs_tracks_simple_ip], label=['Tracks','fs_Tracks','fs_Simple_IP'])
    plt.legend(loc='upper right')
    plt.title('Histogram for the '+ name +' of all Tracks for different types of tracks')
    plt.savefig(name +' of All Tracks.png')
    plt.cla()
'''

#For adjusting the cosmetics of each attribute of the tracks
'''
var_tracks = tracks['d0Uncertainty'].flatten()
var_fs_tracks = fs_tracks['d0Uncertainty'].flatten()
var_fs_tracks_simple_ip = fs_tracks_simple_ip['d0Uncertainty'].flatten()
plt.hist([var_tracks,var_fs_tracks,var_fs_tracks_simple_ip], label=['Tracks','fs_Tracks','fs_Simple_IP'], range=[0,2], bins=20)
plt.yscale("log")
plt.legend(loc='upper right')
plt.title('Histogram for the '+ 'd0Uncertainty' +' of all Tracks for different types of tracks')
plt.savefig('d0Uncertainty' +' of All Tracks.png')
plt.cla()
'''

# An algorithm that knows if the d0 histograms is symmetric about zero of not
'''
var_tracks = tracks['d0'].flatten()
above = var_tracks > 0
above = above.astype(int)
below = var_tracks < 0
below = below.astype(int)
if np.sum(above) == np.sum(below): 
    print("The " + "d0" + " histogram is symmetric about zero")
else:
    print("The " + "d0" + " histogram is NOT symmetric about zero")
'''