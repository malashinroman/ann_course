from typing import Sequence, Tuple
import numpy as np


def stretch_gaps(a: np.ndarray, min_gap: int) -> np.ndarray:
    """
    Обеспечивает, чтобы каждый соседний интервал в сортированном массиве был не меньше min_gap

    Args:
        a (np.ndarray): одномерный отсортированный массив целых уровней
        min_gap (int): минимальный допустимый разрыв между соседними элементами

    Returns:
        np.ndarray: новый массив целых чисел, где разрывы >= min_gap
    """
    # создаём копию массива целых чисел
    b: np.ndarray = a.astype(np.int32).copy()
    for i in range(1, b.shape[0]):
        gap: int = int(b[i] - b[i - 1])
        if gap < min_gap:
            shift: int = min_gap - gap
            # сдвигаем все последующие элементы на целое число
            b[i:] += shift
    return b


def stretch_between_anchors(
    levels: np.ndarray,
    anchors: Sequence[int],
    min_gap: int
) -> np.ndarray:
    """
    Разбивает отсортированный массив уникальных уровней на сегменты по якорям и применяет stretch_gaps

    Args:
        levels (np.ndarray): сортированный массив уникальных уровней (uint8)
        anchors (Sequence[int]): значения, которые остаются фиксированными
        min_gap (int): минимальный разрыв внутри каждого сегмента

    Returns:
        np.ndarray: массив целых, где сегменты растянуты, а якоря не изменены
    """
    # сортируем и приводим к целым
    L: np.ndarray = np.sort(levels.astype(np.int32))
    anchors_arr: np.ndarray = np.sort(np.array(anchors, dtype=np.int32))

    # находим индексы якорей
    mask: np.ndarray = np.isin(L, anchors_arr)
    idxs: Sequence[int] = np.nonzero(mask)[0].tolist()

    # формируем сегменты [start, end]
    segments: list[Tuple[int, int]] = []
    prev: int = 0
    for idx in idxs:
        segments.append((prev, idx))
        prev = idx
    segments.append((prev, L.size - 1))

    # копия для новых значений
    L_new: np.ndarray = L.copy()
    for start, end in segments:
        # растягиваем локальный сегмент
        L_new[start : end + 1] = stretch_gaps(L[start : end + 1], min_gap)

    return L_new


def build_lookup(
    levels_orig: np.ndarray,
    levels_new: np.ndarray
) -> np.ndarray:
    """
    Строит таблицу соответствий (LUT) uint8 для преобразования уровней

    Args:
        levels_orig (np.ndarray): исходные уровни (0–255)
        levels_new (np.ndarray): новые уровни после растяжения

    Returns:
        np.ndarray: массив uint8 длины 256, где индекс — старое значение, а элемент — новое
    """
    # инициализируем identity LUT
    lut: np.ndarray = np.arange(256, dtype=np.uint8)
    for o, n in zip(levels_orig, levels_new):
        # округляем и ограничиваем [0,255]
        val: int = int(n)
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        lut[int(o)] = val
    return lut


def stretch_image(
    img: np.ndarray,
    anchors: Tuple[int, ...] = (0, 255),
    min_gap: int = 5
) -> np.ndarray:
    """
    Растягивает плотные кластеры значений в 8-битном изображении, фиксируя якоря

    Args:
        img (np.ndarray): одно­канальное изображение dtype=uint8
        anchors (Tuple[int, ...], optional): уровни, которые не изменяются
        min_gap (int, optional): минимальный разрыв после растяжения

    Returns:
        np.ndarray: новое изображение uint8 с теми же формой и фиксацией якорей
    """
    # 1) уникальные уровни
    levels: np.ndarray = np.unique(img)
    # 2) новые позиции уровней (целые)
    levels_new: np.ndarray = stretch_between_anchors(levels, anchors, min_gap)
    # 3) строим LUT
    lut: np.ndarray = build_lookup(levels, levels_new)
    # 4) применяем LUT к изображению
    return lut[img]
