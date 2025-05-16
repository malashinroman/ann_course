import zipfile
import requests
from tqdm import tqdm
from pathlib import Path


def download_from_zenodo(
    url: str,
    target_dir: Path,
    filename: str,
    force: bool = False
) -> Path:
    """
    Скачивает архив filename в папку target_dir, если его там нет или force=True.

    Args:
        url (str): Прямая ссылка на скачивание с Zenodo.
        target_dir (Path): Папка для сохранения архива.
        filename (str): Имя файла архива (например, 'glioma_mini.zip').
        force (bool): Если True, всегда перекачивает архив.

    Returns:
        Path: Путь к скачанному архиву.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / filename

    if archive_path.exists() and not force:
        return archive_path

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(archive_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f"Downloading {filename}"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                written = f.write(chunk)
                progress_bar.update(written)

    return archive_path


def extract_zip(
    archive_path: Path,
    extract_dir: Path
) -> None:
    """
    Распаковывает zip-архив archive_path в папку extract_dir.

    Args:
        archive_path (Path): Путь к zip-файлу.
        extract_dir (Path): Папка для распаковки содержимого.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(path=extract_dir)


def ensure_glioma_mini(
    zenodo_url: str = "https://zenodo.org/records/15395623/files/glioma_mini.zip?download=1",
    root_dir: Path = Path('.'),
    archive_name: str = 'glioma_mini.zip',
    force_download: bool = False
) -> Path:
    """
    Обеспечивает загрузку и распаковку архива glioma_mini.zip в структуру:
      my_project/
      └── glioma_mini/
          ├── glioma_mini.zip
          └── data/
              ├── train/
              └── valid/

    Args:
        zenodo_url (str): Прямая ссылка Zenodo для скачивания архива.
        root_dir (Path): Корневая папка проекта (по умолчанию текущая).
        archive_name (str): Имя архива (по умолчанию 'glioma_mini.zip').
        force_download (bool): Если True, перекачивает архив даже если он уже есть.

    Returns:
        Path: Путь к папке data/ внутри glioma_mini.
    """
    glioma_root = root_dir / 'glioma_mini'
    archive_path = download_from_zenodo(
        url=zenodo_url,
        target_dir=glioma_root,
        filename=archive_name,
        force=force_download
    )

    data_dir = glioma_root / 'data'
    extract_zip(archive_path, data_dir)

    # Проверка наличия нужных папок
    assert (data_dir / 'train').is_dir(), f"Не найдена папка: {data_dir / 'train'}"
    assert (data_dir / 'valid').is_dir(), f"Не найдена папка: {data_dir / 'valid'}"

    return data_dir