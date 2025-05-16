import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms, datasets
import numpy as np

import os
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List


def get_data_loaders(batch_size: int, data_dir: str = "data") -> Tuple[DataLoader, DataLoader]:
    """Подготавливает загрузчики MNIST для обучения и тестирования."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super().__init__()
        # Построение последовательности слоев (MLP)
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        # Добавляем выходной линейный слой
        layers.append(nn.Linear(prev_dim, output_dim, bias=True))
        self.network = nn.Sequential(*layers)
        
        # Инициализация весов после создания сети
        self.network.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        1) Выпрямляет входные изображения до вектора размерности (batch_size, input_dim)
        2) Пропускает через MLP
        """
        # Вытягивает все измерения кроме batch
        x_flat = x.view(x.size(0), -1)
        return self.network(x_flat)

    def _init_weights(self, module: nn.Module) -> None:
        """Инициализация весов для линейных слоев"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    data_loader_type: str,
    device: torch.device
) -> float:
    """
    Оценивает точность классификации модели на данных из data_loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluation [{data_loader_type}]", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    num_epochs: int
) -> None:
    """
    Тренирует модель и логгирует результаты функции потерь 
    и точностей на обучающей и валидационных выборках.
    """
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"runs/ep{num_epochs}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = float(np.mean(epoch_losses))

        # Оценка после каждой эпохи
        train_acc = compute_accuracy(model, train_loader, "Train", device)
        valid_acc = compute_accuracy(model, valid_loader, "Valid",device)
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)
        print(f"Epoch {epoch:02d}/{num_epochs} - Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение будет на [{device}]")

    # --- Настройки сети и данных ---
    batch_size = 100  # размер мини-батча; увеличьте для большей скорости, уменьшите при нехватке памяти

    train_loader, valid_loader = get_data_loaders(batch_size=batch_size, data_dir='mnist')

    # Размер входного вектора: 28x28 пикселей в MNIST
    input_size = 28 * 28  # 784 входных признаков
    # Размеры скрытых слоев: первый 64 нейронов, второй 32, далее 25 слоев по 16
    hidden_sizes = [64, 32] + [16] * 25  # менять можно всё, кроме числа 25
    # Количество выходных классов: 10 цифр в MNIST
    output_size = 10

    model = MLPClassifier(input_size, hidden_sizes, output_size).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    loss_fn = nn.CrossEntropyLoss()

    train(
        model=model, 
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=10
    )
