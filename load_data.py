from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split


def _split_train_dataset(train_dataset, validation_split, seed):
    """
    Split the original training dataset into training and validation subsets.

    The split is deterministic for a given seed, which makes experiments
    reproducible and ensures that models compared with the same seed use
    exactly the same validation samples.
    """
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1).")

    if validation_split == 0.0:
        return train_dataset, None

    validation_size = int(len(train_dataset) * validation_split)
    train_size = len(train_dataset) - validation_size

    if validation_size == 0 or train_size == 0:
        raise ValueError(
            "validation_split is too small or too large for the dataset."
        )

    generator = torch.Generator().manual_seed(seed)

    train_subset, validation_subset = random_split(
        train_dataset,
        [train_size, validation_size],
        generator=generator
    )

    return train_subset, validation_subset


def _build_loaders(train_dataset, test_dataset, batch_size,
                   validation_split=0.1, seed=0):
    """
    Build train, validation and test DataLoaders.

    validation_split is the fraction of the original training set reserved
    for validation. The test set is never split and remains untouched.
    """
    train_dataset, validation_dataset = _split_train_dataset(
        train_dataset,
        validation_split,
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader, test_loader


def load_MNIST_fashion(batch_size, validation_split=0.1, seed=0):

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return _build_loaders(
        train_dataset,
        test_dataset,
        batch_size,
        validation_split,
        seed
    )


def load_CIFAR_100(batch_size, validation_split=0.1, seed=0):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return _build_loaders(
        train_dataset,
        test_dataset,
        batch_size,
        validation_split,
        seed
    )


def load_CIFAR_10(batch_size, validation_split=0.1, seed=0):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return _build_loaders(
        train_dataset,
        test_dataset,
        batch_size,
        validation_split,
        seed
    )
