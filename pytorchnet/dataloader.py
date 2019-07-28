import numpy as np


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ExampleData(Dataset):
    def __init__(self, data_shape, num):
        self.num = num
        self.data_shape = data_shape
        self.data, self.label = self.gen_data(self.num, self.data_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        sample = {'input': self.preprocess(data), 'label': label}
        return sample

    def gen_data(self, num, data_shape):
        np.random.seed(19)
        data = np.random.randn(num, *data_shape)
        label = np.random.randint(0, 3, num)
        return data, label

    def preprocess(self, data):
        return data