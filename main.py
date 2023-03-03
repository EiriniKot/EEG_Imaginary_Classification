"""
EEG data, per subject, per event.
Splitted data from Data_filtered_ICA folder which contains ICA cleared data.
For each event the T_start is 0.2 prior to stimulus, i.e. T_start = -0.2 and T_end = 0.5
The total duration of each event is 0.7 seconds = 359 time points
The total number of channels in the final configuration is 69.
The shape of the array of each .npy file is (69,359).
"""

import os, json
from src.preprocessing import dataset_to_torch_save, apply_stft, split_ttv, apply_cwt
from src.utils import run_train_nn, dataset_loaders
from src.nn_net import Net
import torch
from sklearn.svm import SVC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Change this path into your data path!
dataset = '/home/eirini/Documents/biomedical/Imaginary_vis/session2/'
label_categories = os.listdir(dataset)
f = open('onehot_info.json')
label_index_info = json.load(f)

save = False
if save:
    # Call this function to create torch initial datasets per label with one hot encoding
    dataset_to_torch_save(dataset, label_categories, label_index_info)

stft = False
if stft:
    sets_path = [os.path.join('torch_data', dt) for dt in os.listdir('torch_data')]
    o = apply_stft(sets_path, return_tensor=True, save=True)

cwt = True
if cwt:
    """
    Performs a continuous wavelet transform on data, using the wavelet function.
    A CWT performs a convolution with data using the wavelet function, which is
    characterized by a width parameter and length parameter. """
    sets_path = [os.path.join('torch_data', dt) for dt in os.listdir('torch_data')]
    apply_cwt(sets_path)

split = True
if split:
    sets_path = [os.path.join('torch_cwt', dt) for dt in os.listdir('torch_cwt')]
    split_ttv(sets_path, folder='torch_split', train=0.7, val=0.15)

train = True
if train:
    sets_path = [os.path.join('torch_split', dt) for dt in os.listdir('torch_split')]
    loaders = dataset_loaders(paths=sets_path, batch_size=12, shuffle=True)

    network = Net(c=63, d=9, h=359, outputs=4).to(device)

    opt = torch.optim.Adam(network.parameters(), lr=0.0005)
    run_train_nn(datasets_loaders=loaders,
                 network=network,
                 optimizer=opt,
                 epochs=100,
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))