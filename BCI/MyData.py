from torch.utils.data import Dataset


# 单模态数据
class UniModalDataset(Dataset):  # 需要继承data.Dataset

    def __init__(self, data, label):
        # TODO
        # 1. Initialize file path or list of file names.

        self.data = data
        self.label = label
        self.len = label.shape[0]

        print("Dataset.shape:")
        print("  data.shape: ", data.shape)
        print("  label.shape: ", label.shape)
        print("\n")

        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        data = self.data[index, ...]
        label = self.label[index]

        return data, label

        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset.

        return self.len


# EEG数据+NIRS数据
class MultiModalDataset(Dataset):  # 需要继承data.Dataset

    def __init__(self, EEGdata, NIRSdata, label):
        # TODO
        # 1. Initialize file path or list of file names.

        self.EEGdata = EEGdata
        self.NIRSdata = NIRSdata
        self.label = label
        self.len = label.shape[0]

        print("Dataset.shape:")
        print("  EEGdata.shape: ", EEGdata.shape)
        print("  NIRSdata.shape: ", NIRSdata.shape)
        print("  label.shape: ", label.shape)
        print("\n")

        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        EEGdata = self.EEGdata[index, ...]
        NIRSdata = self.NIRSdata[index, ...]
        label = self.label[index]

        return EEGdata, NIRSdata, label

        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset.

        return self.len




