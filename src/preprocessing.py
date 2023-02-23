import numpy as np
import os
import torch

from scipy.signal import cwt, ricker


def dataset_to_torch_save(dataset, label_categories, label_info):
    """
    This function is used in order to load data and save them as torch dataset
    Each label is going to have a different dataset_xx.pt file saved in torch_data folder
    :return:
    """
    for label in label_categories:
        full_label_path = os.path.join(dataset, label)
        arrays_list = list(map(lambda smpl_name: np.load(os.path.join(full_label_path, smpl_name)),
                               os.listdir(full_label_path)))

        arr_x = torch.tensor(arrays_list)
        num_examples = arr_x.shape[0]
        one_pos = label_info[label]
        arr_y = torch.zeros((num_examples, len(label_categories)))
        arr_y[:, one_pos] = 1
        print(f'Making of label {label}')
        dataset_torch = torch.utils.data.TensorDataset(arr_x, arr_y)
        torch.save(dataset_torch, f'torch_data/dataset_{label}.pt')


def apply_stft(sets_path,
               return_tensor=True,
               save=True):

    if return_tensor:
        X = []
        y = []

    for set in sets_path:
        torch_set = torch.load(set)
        input_x = []
        input_y = []
        for input_s, label in torch_set:
            # Computes the 2 dimensional discrete Fourier transform of input. Equivalent to fftn()
            # but FFTs only the last two dimensions by default.
            input_fft = torch.stft(input_s, n_fft=100, center=False, hop_length=6)
            # ?*Number Frequencies * Total Frames *2
            input_x.append(input_fft[:,:,:,0])
            input_y.append(label)

        final = torch.stack(input_x)
        label = torch.stack(input_y)

        out = set.split('_')[-1]
        if save:
            dataset_torch = torch.utils.data.TensorDataset(final, label)
            torch.save(dataset_torch, f'torch_stft/dataset_stft_{out}')

        if return_tensor:
            X.append(final)
            y.append(label)

    if return_tensor:
        X = torch.cat(X)
        y = torch.cat(y)
        return X, y


def apply_cwt(sets_path, save=True):
    for set in sets_path:
        torch_set = torch.load(set)
        input_x = []
        input_y = []
        widths = np.arange(1, 10)
        for input_s, label in torch_set:
            cwt_per_c = []
            for each_channel in input_s:
                cwtmatr = cwt(each_channel, ricker, widths)
                cwt_per_c.append(cwtmatr)
            cwt_all_channels_stacked = np.stack(cwt_per_c, 0)
            input_x.append(torch.from_numpy(cwt_all_channels_stacked))
            input_y.append(label)

        final = torch.stack(input_x)
        label = torch.stack(input_y)

        out = set.split('_')[-1]
        if save:
            dataset_torch = torch.utils.data.TensorDataset(final, label)
            torch.save(dataset_torch, f'torch_cwt/dataset_cwt_{out}')


def split_ttv(sets_path, folder = 'torch_split', train=0.7, val=0.15):
    for set in sets_path:
        torch_set = torch.load(set)
        iterator = torch.utils.data.DataLoader(torch_set, batch_size=1, shuffle=True)
        shuffled_X = []
        shuffled_y = []

        n_train_s = int(train*len(torch_set))
        n_valid_s = int(val * len(torch_set))

        for input_s, label in iterator:
            shuffled_X.append(input_s)
            shuffled_y.append(label)

        train_X = shuffled_X[:n_train_s]
        train_y = shuffled_y[:n_train_s]

        out = set.split('_')[-1]

        train_X = torch.cat(train_X, 0)
        train_y = torch.cat(train_y, 0)

        dataset_torch = torch.utils.data.TensorDataset(train_X, train_y)
        torch.save(dataset_torch, f'{folder}/dataset_split_train_{out}')

        del dataset_torch, train_X, train_y

        val_X = shuffled_X[n_train_s:n_train_s + n_valid_s]
        val_y = shuffled_y[n_train_s:n_train_s + n_valid_s]

        val_X = torch.cat(val_X, 0)
        val_y = torch.cat(val_y, 0)

        dataset_torch = torch.utils.data.TensorDataset(val_X, val_y)
        torch.save(dataset_torch, f'{folder}/dataset_split_val_{out}')

        del val_X, val_y, dataset_torch

        test_X = shuffled_X[n_train_s + n_valid_s:]
        test_y = shuffled_y[n_train_s + n_valid_s:]

        test_X = torch.cat(test_X, 0)
        test_y = torch.cat(test_y, 0)

        dataset_torch = torch.utils.data.TensorDataset(test_X, test_y)
        torch.save(dataset_torch, f'{folder}/dataset_split_test_{out}')

        del test_X, test_y, dataset_torch



