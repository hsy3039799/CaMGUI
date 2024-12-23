import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Double_dataset(Dataset):
    def __init__(self, data, targets, transform_fixmatch):
        self.data = data
        self.targets = targets
        self.n = len(targets)
        self.index = list(range(self.n))
        self.transform_fixmatch = transform_fixmatch

    def __getitem__(self, i):
        img = self.data[i]
        # 确保输入数据为PIL图像，因为TransformFixMatch_CIFAR10可能期望PIL图像输入
        img = Image.fromarray(img)

        # 应用弱增强和强增强变换
        img_weak = self.transform_fixmatch.weak(img)
        img_strong = self.transform_fixmatch.strong(img)

        return img_weak, img_strong, self.targets[i], self.index[i]

    def __len__(self):
        return len(self.index)

    def update_label(self, noise_label):
        self.targets[:] = noise_label[:]

        
class Custom_dataset(Dataset):
    """
    Custom dataset class for handling data and targets.

    Attributes:
    - data: The input data.
    - targets: The labels for the data.
    - n: The number of data points.
    - index: A list of indices for the data points.
    - transform: The transformation to apply to the data.
    """
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data, targets, transform=transform_test):
        """
        Initialize the dataset with data, targets, and an optional transform.

        Parameters:
        - data: The input data.
        - targets: The labels for the data.
        - transform: The transformation to apply to the data (default is transform_test).
        """
        self.data = data
        self.targets = targets
        self.n = len(list(targets))
        self.index = list(range(self.n))
        self.transform = transform

    def __getitem__(self, i):
        """
        Get the i-th data point and its label.

        Parameters:
        - i: The index of the data point.

        Returns:
        - img: The transformed image.
        - self.targets[i]: The label of the image.
        - self.index[i]: The index of the image.
        """
        img = self.data[i]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i], self.index[i]

    def __len__(self):
        """
        Return the number of data points in the dataset.

        Returns:
        - self.n: The number of data points.
        """
        return self.n

    def update_label(self, noise_label):
        """
        Update the labels with noisy labels.

        Parameters:
        - noise_label: The new noisy labels.
        """
        self.targets[:] = noise_label[:]


def get_dataset(dataroot):
    """
    Get the CIFAR-10 training and test datasets.

    Parameters:
    - dataroot: The root directory where the datasets are stored.

    Returns:
    - train_dataset: The CIFAR-10 training dataset.
    - test_dataset: The CIFAR-10 test dataset.
    """
    train_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True)

    return train_dataset, test_dataset
