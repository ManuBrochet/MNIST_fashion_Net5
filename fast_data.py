"""
GPU-resident data pipeline (variant "c").

The datasets used here are 47-225 MiB as uint8 and fit trivially in the 11 GB of
a 1080 Ti. Holding them on the device removes the per-image PIL/ToTensor/Normalize
work that dominated every batch, and removes the host->device copy entirely.

Storing uint8 is NOT quantization: that is the native dtype of the source data,
and widening to float32 is exact. Normalisation runs on the GPU, which differs
from the CPU pipeline by up to 2 ULP of float32 - see notes in the playbook.
"""
import torch

_STATS = {                      # (mean, std) or None to skip normalisation
    "CIFAR10":       ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "CIFAR100":      ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "SVHN":          ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "MNIST_fashion": None,
}


class GPULoader:
    """Iterable over (images, labels) batches already on `device`.

    Mirrors just enough of the DataLoader interface for the training loop:
    iteration and len(). `shuffle` and `drop_last` follow the DataLoader
    semantics they replace.
    """

    def __init__(self, x_u8, y, batch_size, mean, std,
                 shuffle=False, drop_last=False, generator=None):
        self.x, self.y = x_u8, y
        self.bs, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.mean, self.std, self.generator = mean, std, generator

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.y)
        if self.shuffle:
            idx = torch.randperm(n, device=self.x.device, generator=self.generator)
        else:
            idx = torch.arange(n, device=self.x.device)
        last = n - self.bs + 1 if self.drop_last else n
        for i in range(0, last, self.bs):
            j = idx[i:i + self.bs]
            images = self.x[j].float().div_(255)
            if self.mean is not None:
                images = images.sub_(self.mean).div_(self.std)
            yield images, self.y[j]


def _raw(dataset):
    """uint8 NCHW tensor + int64 labels from a torchvision dataset."""
    data = dataset.data
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = dataset.targets
    x = torch.as_tensor(data)
    if x.ndim == 3:                       # (N,H,W) grayscale -> (N,1,H,W)
        x = x.unsqueeze(1)
    elif x.shape[-1] in (1, 3):           # (N,H,W,C) -> (N,C,H,W)
        x = x.permute(0, 3, 1, 2)
    return x.contiguous(), torch.as_tensor(labels).long()


def build_loaders(train_dataset, test_dataset, dataset_name, batch_size,
                  validation_split=0.1, seed=0, device=None):
    """Drop-in replacement for load_data._build_loaders.

    The train/validation split uses the same CPU generator and the same
    random_split index semantics as the original, so a given seed selects the
    same validation samples as before.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = _STATS.get(dataset_name)
    mean = std = None
    if stats is not None:
        mean = torch.tensor(stats[0], device=device).view(1, -1, 1, 1)
        std = torch.tensor(stats[1], device=device).view(1, -1, 1, 1)

    xtr, ytr = _raw(train_dataset)
    xte, yte = _raw(test_dataset)
    xtr, ytr = xtr.to(device), ytr.to(device)
    xte, yte = xte.to(device), yte.to(device)

    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1).")

    val_loader = None
    if validation_split > 0.0:
        n_val = int(len(ytr) * validation_split)
        if n_val == 0 or n_val == len(ytr):
            raise ValueError("validation_split is too small or too large.")
        # torch.randperm on CPU with a seeded generator reproduces the index
        # order random_split(generator=...) would have produced.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ytr), generator=g).to(device)
        tr_idx, val_idx = perm[: len(ytr) - n_val], perm[len(ytr) - n_val:]
        val_loader = GPULoader(xtr[val_idx], ytr[val_idx], batch_size, mean, std)
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]

    g_dev = torch.Generator(device=device); g_dev.manual_seed(seed)
    train_loader = GPULoader(xtr, ytr, batch_size, mean, std,
                             shuffle=True, drop_last=True, generator=g_dev)
    test_loader = GPULoader(xte, yte, batch_size, mean, std)
    return train_loader, val_loader, test_loader
