import torch
from torch.utils.data import DataLoader, ConcatDataset


def dataset_loaders(paths: list,
                    batch_size: int = 10,
                    shuffle: bool = True,
                    ):
    test_sets = []
    train_sets = []
    valid_sets = []

    for path in paths:
        if 'test' in path:
            test_sets.append(torch.load(path))
        elif 'train' in path:
            train_sets.append(torch.load(path))
        elif 'val' in path:
            valid_sets.append(torch.load(path))

    train_set = ConcatDataset(train_sets)
    train_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_set = ConcatDataset(test_sets)
    test_set_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    valid_set = ConcatDataset(valid_sets)
    valid_set_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)

    return {'train_set_loader': train_set_loader,
            'test_set_loader': test_set_loader,
            'valid_set_loader': valid_set_loader}


def run_train_nn(datasets_loaders:dict, network, optimizer, epochs, loss_fn, device):
    for ep in range(epochs):
        t_loss, v_loss, v_acc, t_acc = train_one_epoch(datasets_loaders['train_set_loader'],
                                         datasets_loaders['valid_set_loader'],
                                         optimizer, network, loss_fn, print_step=10,
                                         device = device)
        print(f'Epoch number {ep} ---> train loss: {round(t_loss,4)}  train accuracy: {t_acc} val loss: {round(v_loss,4)} val accuracy: {v_acc}')
        print('\n')


def train_one_epoch(training_loader, val_loader, optimizer, network, loss_fn, print_step=10,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    running_loss = 0.
    last_loss = 0.
    tr_acc = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair

        inputs, labels = data

        inputs.to(device)
        labels.to(device)

        optimizer.zero_grad()
        outputs = network(inputs)
        # Compute the loss and its gradients

        loss = loss_fn(outputs, labels)
        correct_indexes = torch.argmax(labels, 1)
        prect_indexes = torch.argmax(outputs, 1)
        tr_acc += float(torch.sum(correct_indexes == prect_indexes)) / len(correct_indexes)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % print_step == print_step-1:
            last_loss = float(running_loss / print_step)  # loss per batch
            print(f'Per {print_step} steps batch {i + 1} loss: {round(last_loss,4)}')
            running_loss = 0.
    tr_acc = float(tr_acc / len(training_loader))

    val_acc = 0.
    running_loss = 0.
    with torch.no_grad():

        for i, data in enumerate(val_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            outputs = network(inputs)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            correct_indexes = torch.argmax(labels ,1)
            prect_indexes = torch.argmax(outputs, 1)
            val_acc += float(torch.sum(correct_indexes == prect_indexes))/len(correct_indexes)
            # Gather data and report
            running_loss += loss.item()

    val_loss = float(running_loss/len(val_loader))
    val_acc = float(val_acc/len(val_loader))
    # print(f'Validation loss is {round(val_loss, 4)}')

    return last_loss, val_loss, val_acc, tr_acc
