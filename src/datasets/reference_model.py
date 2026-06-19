#clean reference model for atypicality-based edge tail selection
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.config import CIFAR10_MEAN, CIFAR10_STD, REFERENCE_MODEL_PATH
from src.models.model import build_model


#pick best available device
def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#train clean resnet on full cifar then save weights
def build_reference_model(path=REFERENCE_MODEL_PATH, epochs=15, lr=0.05, seed=42):
    torch.manual_seed(seed)
    device = _device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train, batch_size=128, shuffle=True)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        print(f"reference epoch {ep + 1}/{epochs} done")

    torch.save(model.state_dict(), path)
    print(f"saved reference model to {path}")


#load reference model for tail selection
def load_reference_model(path=REFERENCE_MODEL_PATH):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    build_reference_model()
