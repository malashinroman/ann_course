from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# Определяем устройство вычислений (GPU, если доступен)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    """Сверточная нейронная сеть для CIFAR-10.

    Архитектура:
      1) Conv2d(3->32,5) + ReLU + MaxPool2d(2)
      2) Conv2d(32->64,5) + ReLU + MaxPool2d(2)
      3) Flatten -> Linear(64*5*5->120) + ReLU
      4) Linear(120->10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Метод прямого прохода сети.

        Args:
            x (torch.Tensor): Тензор [N,3,32,32]

        Returns:
            torch.Tensor: Логиты [N,10]

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_name(self) -> str:
        """Возвращает 'conv'."""
        return "conv"


class LinearNet(nn.Module):
    """Полносвязная сеть для CIFAR-10.

    Архитектура:
      1) Flatten -> Linear(3*32*32->320) + Sigmoid
      2) Linear(320->120) + Sigmoid
      3) Linear(120->84) + Sigmoid
      4) Linear(84->10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 320)
        self.fc2 = nn.Linear(320, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Метод прямого прохода сети.

        Args:
            x (torch.Tensor): Тензор [N,3,32,32]

        Returns:
            torch.Tensor: Логиты [N,10]

        """
        x = x.view(-1, 32 * 32 * 3)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return self.fc4(x)

    def get_name(self) -> str:
        """Возвращает 'linear'."""
        return "linear"


def get_data_loaders(
    batch_size: int = 50,
    train_subset_size: int = 10000,
    data_dir: str = "data",
) -> tuple[DataLoader, DataLoader]:
    """Создает DataLoader для CIFAR-10.

    Args:
        batch_size (int): Размер батча
        train_subset_size (int): Размер обучающей выборки
        data_dir (str): Путь к папке для загрузки набора данных
    Returns:
        train_loader, test_loader

    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_ds = torch.utils.data.Subset(train_ds, list(range(train_subset_size)))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_ds, batch_size=50, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def show_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    """Вытягивает батч в сетку, декодирует и показывает картинку,
        а под ней печатает соответствующие метки.

    Args:
        images (torch.Tensor): Батч изображений
        labels (torch.Tensor): Батч меток

    """
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Собираем сетку изображений (B x C x H x W -> C x (H*rows) x (W*cols))
    grid = torchvision.utils.make_grid(images)
    # «Декодируем» нормализованные тензоры
    grid = grid / 2 + 0.5  # при нормализации mean=0.5, std=0.5
    np_img = grid.cpu().numpy()  # C x H' x W'

    # Печатаем имена классов
    print(
        f"Классы для текущего батча ({images.size(0)}):",
        " ".join(classes[int(l)] for l in labels),
        "\n",
    )

    # Переставляем каналы и показываем
    plt.imshow(np_img.transpose(1, 2, 0))
    plt.axis("off")
    plt.show()


def calculate_batch_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int]:
    """Вычисляет точность для одного батча.

    Args:
        outputs (torch.Tensor): Логиты модели, форма [B, C], где B —
            размер батча, C — число классов
        labels (torch.Tensor): Истинные метки, форма [B]

    Returns:
        tuple(int, int): Количество правильных предсказаний и размер батча

    """
    # Число элементов в батче
    batch_size = labels.size(0)
    # Индексы максимальных логитов -> предсказанные классы
    _, preds = torch.max(outputs, dim=1)
    # Сравниваем с истинными метками и считаем корректные
    correct = (preds == labels).sum().item()
    return int(correct), batch_size


def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int = 10,
) -> None:
    """Обучает модель по эпохам и логирует метрики в TensorBoard.

    Args:
        model (nn.Module): Модель, перенесенная на DEVICE
        loss_fn (nn.Module): Функция потерь
        optimizer (Optimizer): Оптимизатор
        train_loader (DataLoader): Загрузчик обучающих данных
        valid_loader (DataLoader): Загрузчик валидационных данных
        num_epochs (int): Число эпох обучения

    """
    # Параметры датасетов
    train_size = len(train_loader.dataset)
    valid_size = len(valid_loader.dataset)
    # Директория логов с метками времени и размерами датасетов
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mname = model.get_name()
    log_dir = (
        f"runs/{mname}_ep{num_epochs}_train{train_size}_test{valid_size}_{ts}"
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    model.to(DEVICE)
    print(
        f"Начинаю обучение: модель={mname}, train={train_size}, "
        f"valid={valid_size}, device={DEVICE.type}"
    )

    for epoch in range(1, num_epochs + 1):
        # ===== Тренировочный цикл =====
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"
        ):
            # Перемещение на DEVICE
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            # Вычисление потерь и шаг оптимизации
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # Накопление метрик по батчу
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size  # суммарная потеря
            correct, _ = calculate_batch_accuracy(outputs, labels)
            total_correct += correct
            total_samples += batch_size
        # Средняя потеря и точность на train
        avg_train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ===== Валидационный цикл =====
        model.eval()
        val_correct, val_samples = 0, 0
        for imgs, labels in tqdm(
            valid_loader,
            desc=f"Epoch {epoch}/{num_epochs} [Valid]",
            leave=False,
        ):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                outputs = model(imgs)
            correct, _ = calculate_batch_accuracy(outputs, labels)
            val_correct += correct
            val_samples += labels.size(0)
        valid_acc = val_correct / val_samples  # точность на valid

        # Логируем в TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        # Печать результатов эпохи
        print(
            f"Epoch {epoch}/{num_epochs} - loss: {avg_train_loss:.4f}, "
            f"train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}"
        )

    writer.close()
    print("Обучение завершено. Логи сохранены в:", log_dir)


if __name__ == "__main__":
    model_type = "conv"
    train_loader, valid_loader = get_data_loaders(
        batch_size=50, train_subset_size=10000, data_dir="cifar10"
    )

    model = ConvNet() if model_type == "conv" else LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    show_batch(images, labels)

    train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=40,
    )
