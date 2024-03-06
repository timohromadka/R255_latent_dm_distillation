from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class GeneralDataModule(LightningDataModule):
    def __init__(self, data_dir: str, dataset_name: str, batch_size: int, image_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage=None):
        if self.dataset_name == 'mnist':
            self.dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        elif self.dataset_name == 'cifar10':
            self.dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)

    @property
    def transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if self.dataset_name == 'mnist' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)