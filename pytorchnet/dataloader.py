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

    def gen_data(self, num, data_shape, seed=19):
        np.random.seed(seed)
        data = np.random.randint(0, 255, (num, *data_shape), dtype=np.int32)
        # data = np.random.randint(1, 5, (num, data_shape))
        np.random.seed(seed)
        label = np.random.randint(0, 3, num)
        # data = np.zeros((num, *data_shape))
        return data, label

    def preprocess(self, data):
        return data