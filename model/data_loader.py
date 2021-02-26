# Import torch libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# Importing datascience helpers
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle


class StumbleDataSet(Dataset):
    def __init__(self):
        # data loading
        self.dset = pickle.load(open("experiments/dataset.pkl", "rb"))
        self.scaler = MinMaxScaler()
        self.x_raw = self.dset[:, :-1].astype('float32')

        self.x = torch.from_numpy(
                self.scaler.fit_transform(self.x_raw)
            )

        self.targetscaler = MinMaxScaler()
        self.y = self.dset[:, [-1]].astype('int')
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        pickle.dump(
                self.scaler,
                open("./experiments/input_scaler.pkl", "wb")
            )

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


class Data_Loaders():

    def __init__(self, batch_size):
        self.nav_dataset = StumbleDataSet()
        self.features = self.nav_dataset.n_features
        shuffle_dataset = True
        random_seed = 42
        dataset_size = self.nav_dataset.n_samples
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(
                self.nav_dataset,
                batch_size=batch_size,
                sampler=train_sampler
            )
        self.test_loader = DataLoader(
                self.nav_dataset,
                batch_size=batch_size,
                sampler=valid_sampler
            )


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # note this is how the dataloaders will be iterated over to test
    for idx, (inputs, labels) in enumerate(data_loaders.train_loader):
        _, _ = inputs, labels
    for idx, (inputs, labels) in enumerate(data_loaders.test_loader):
        _, _ = inputs, labels


if __name__ == '__main__':
    main()
