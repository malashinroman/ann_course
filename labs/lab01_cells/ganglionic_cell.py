from __future__ import annotations
import os
from typing import Dict, Literal, Tuple, List, Union, Optional

import cv2                                              # для обработки изображений
import numpy as np                                      # для численных вычислений
import matplotlib.pyplot as plt                         # для визуализации результатов
from tqdm import tqdm


# Константы ориентации
ORIENTATION_VERTICAL = 'vertical'
ORIENTATION_HORIZONTAL = 'horizontal'

CellTypes = List[Literal["ganglion", "simple", "complex"]]



class ReceptiveField:
    """
    Базовый класс рецептивного поля с позицией и размером.
    """
    def __init__(
        self,
        position: Tuple[int, int],
        size: Tuple[int, int]
    ) -> None:
        """
        Args:
            position (Tuple[int, int]): координаты центра поля
            size (Tuple[int, int]): размер поля в клетках
        Returns:
            None
        """
        # Позиция хранится в виде массива [x, y]
        self.position: np.ndarray = np.array(position, dtype=np.int16)
        # Размер хранится в виде массива [ширина, высота]
        self.size: np.ndarray = np.array(size, dtype=np.int16)

    def get_response(self, image: np.ndarray) -> float:
        """
        Args:
            image (np.ndarray): входное изображение в оттенках серого
        Returns:
            float: базовый отклик рецептивного поля
                   (переопределяется в наследниках)
        """
        # Базовый класс не реализует метод
        raise NotImplementedError("get_response необходимо переопределить в подклассе")

    def set_position(
        self,
        new_position: Union[Tuple[int, int], np.ndarray]
    ) -> None:
        """
        Args:
            new_position (Union[Tuple[int, int], np.ndarray]): новая позиция центра поля
        Returns:
            None
        """
        # Обновляем координаты центра
        self.position = np.array(new_position, dtype=np.int16)


class GanglionCell(ReceptiveField):
    def __init__(
        self,
        position: Tuple[int, int] = (0, 0),
        inner_radius: int = 5,
        outer_radius: int = 11,
        is_off_center: bool = False
    ) -> None:
        """
        Args:
            position (Tuple[int, int], optional): положение центра клетки
            inner_radius (int, optional): радиус внутреннего гауссова ядра
            outer_radius (int, optional): радиус внешнего гауссова ядра
            is_off_center (bool, optional): тип клетки (on-center или off-center)
        Returns:
            None
        """
        # Инициализируем базовый класс с размером 1x1 (одна клетка)
        super().__init__(position, (1, 1))
        self.inner_radius: int = inner_radius
        self.outer_radius: int = outer_radius
        self.is_off_center: bool = is_off_center

    def get_response(self, image: np.ndarray) -> float:
        """
        Args:
            image (np.ndarray): входное изображение в оттенках серого
        Returns:
            float: разностный отклик (он- или офф-центр)
        """
        # Гладим внутреннюю и внешнюю области
        blurred_inner = cv2.GaussianBlur(image, (self.inner_radius, self.inner_radius), sigmaX=0)
        blurred_outer = cv2.GaussianBlur(image, (self.outer_radius, self.outer_radius), sigmaX=0)
        # Вычисляем разницу: on-центр минус off-центр или наоборот
        diff = blurred_outer - blurred_inner if self.is_off_center else blurred_inner - blurred_outer
        x, y = self.position
        # Возвращаем значение отклика в центральной точке
        return float(diff[y, x])


class SimpleCell(ReceptiveField):  # Простая клетка, ансамбль ганглиозных клеток
    def __init__(
        self,
        position: Tuple[int, int] = (0, 0),
        ganglion_layout: Tuple[int, int] = (5, 5),
        span: int = 3,
        is_off_type: bool = False,
        orientation: str = ORIENTATION_VERTICAL
    ) -> None:
        """
        Args:
            position (Tuple[int, int], optional): центральная позиция поля
            ganglion_layout (Tuple[int, int], optional): размер сетки ганглиозных клеток
            span (int, optional): шаг между ними (в пикселях)
            is_off_type (bool, optional): тип клетки on или off
            orientation (str, optional): ориентация (вертикальная/горизонтальная)
        Returns:
            None
        """
        super().__init__(position, ganglion_layout)
        self.span = span
        self.orientation = orientation
        # Гарантируем нечетное число клеток по каждой оси
        self.size[0] += int(not(self.size[0] & 1))
        self.size[1] += int(not(self.size[0] & 1))

        # Создаем сетку ганглиозных клеток
        self.ganglion_cells: List[GanglionCell] = []
        half_x, half_y = self.size[0] // 2, self.size[1] // 2
        for dx in range(-half_x, half_x + 1):
            for dy in range(-half_y, half_y + 1):
                # Решаем, on- или off-клетка
                if is_off_type:
                    cell_type = dx != 0 if self.orientation == ORIENTATION_VERTICAL else dy != 0
                else:
                    cell_type = dx == 0 if self.orientation == ORIENTATION_VERTICAL else dy == 0
                cell_pos = (self.position[0] + dx * self.span, self.position[1] + dy * self.span)
                self.ganglion_cells.append(GanglionCell(position=cell_pos, is_off_center=not cell_type))

    def get_response(self, image: np.ndarray) -> float:
        """
        Args:
            image (np.ndarray): изображение в оттенках серого
        Returns:
            float: сумма откликов всех ганглиозных клеток
        """
        # Суммируем отклики каждого элемента ансамбля
        return sum(cell.get_response(image) for cell in self.ganglion_cells)

    def set_position(
        self,
        new_position: Union[Tuple[int, int], np.ndarray]
    ) -> None:
        """
        Args:
            new_position (Union[Tuple[int, int], np.ndarray]): новая позиция центра поля
        Returns:
            None
        """
        # Вычисляем смещение и применяем к каждому элементу
        new_pos_arr = np.array(new_position, dtype=np.int16)
        delta = new_pos_arr - self.position
        self.position = new_pos_arr
        for cell in self.ganglion_cells:
            cell.position += delta


class ComplexCell(ReceptiveField):
    def __init__(
        self,
        position: Tuple[int, int] = (0, 0),
        simple_layout: Tuple[int, int] = (5, 1),
        span: int = 1,
        ganglion_layout: Tuple[int, int] = (5, 5),
        ganglion_span: int = 3,
        is_off_type: bool = False,
        orientation: str = ORIENTATION_VERTICAL
    ) -> None:
        """
        Инициализация сложной клетки как ансамбля простых клеток.

        Args:
            position (Tuple[int, int], optional): центральная позиция поля.
            simple_layout (Tuple[int, int], optional): сетка простых клеток.
            span (int, optional): шаг между простыми клетками.
            ganglion_layout (Tuple[int, int], optional): сетка внутри простой клетки.
            ganglion_span (int, optional): шаг внутри простой клетки.
            is_off_type (bool, optional): off-тип клетки.
            orientation (str, optional): ориентация ансамбля (вертикальная или горизонтальная).
        """
        # Вызов конструктора родительского класса
        super().__init__(position, simple_layout)

        # Сохраняем параметры
        self.span = span
        self.orientation = orientation

        # Корректируем размер сетки, чтобы она была нечётной
        self.size[0] += int(not (self.size[0] & 1))

        # Создайте список простых клеток и добавьте их в self.simple_cells
        self.simple_cells: List[SimpleCell] = []

        # TODO: реализуйте заполнение self.simple_cells
        # Подсказка:
        # - используйте цикл от -half до +half, где half = self.size[0] // 2
        # - вычислите позицию каждой простой клетки в зависимости от ориентации
        # - создайте SimpleCell с нужными параметрами и добавьте в список
        raise NotImplementedError("Необходимо реализовать создание ансамбля простых клеток.")

    def get_response(self, image: np.ndarray) -> float:
        """
        Возвращает максимальный отклик среди всех простых клеток ансамбля.

        Args:
            image (np.ndarray): входное изображение.
        Returns:
            float: максимальный отклик.
        """
        raise NotImplementedError("Необходимо реализовать метод get_response.")

    def set_position(
        self,
        new_position: Union[Tuple[int, int], np.ndarray]
    ) -> None:
        """
        Сдвигает ансамбль простых клеток в новую позицию.

        Args:
            new_position (Union[Tuple[int, int], np.ndarray]): новая позиция.
        """
        raise NotImplementedError("Необходимо реализовать метод set_position.")


class ReceptiveFieldAnalyzer:
    """
    Класс для тестирования и визуализации реакций разных типов клеток.
    """
    def __init__(
        self,
        ganglion_cell: GanglionCell,
        simple_cell: SimpleCell,
        complex_cell: ComplexCell,
        image_size: Tuple[int, int] = (256, 256)
    ) -> None:
        """
        Args:
            ganglion_span (int, optional): шаг между ганглиозными клетками
            simple_span (int, optional): шаг между простыми клетками
            orientation (str, optional): ориентация клеток
            image_size (Tuple[int, int], optional): размер создаваемого изображения
        Returns:
            None
        """
        # Инициализируем базовые поля
        self.image_size = np.array(image_size, dtype=np.int16)
        self.cells: Dict[str, ReceptiveField] = {
            "ganglion": ganglion_cell,
            "simple":   simple_cell,
            "complex":  complex_cell
        }
        
        # Центрируем все клетки на изображении
        center = tuple(self.image_size // 2)
        for cell in self.cells.values():
            cell.set_position(center)

    def _prepare_image(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: пустое черное изображение для стимулов
        """
        # Создаем изображение, заполненное нулями
        return np.zeros(self.image_size, dtype=np.int16)

    def check_point_response(
        self,
        cell_type: str,
        cell: ReceptiveField,
        field_dims: Tuple[int, int]
    ) -> np.ndarray:
        """
        Args:
            cell_type (str): тип клетки для теста
            cell (ReceptiveField): объект клетки для теста
            field_dims (Tuple[int, int]): размер области точечных стимулов
        Returns:
            np.ndarray: карта откликов на каждый пиксель поля
        """
        field_dims_arr = np.array(field_dims, dtype=np.int16)
        # Расширяем изображение при необходимости для покрытия области теста
        if np.any(self.image_size <= field_dims_arr):
            self.image_size = field_dims_arr + 50
        cell.set_position(tuple(self.image_size // 2))
        
        # Размер результирующей карты для визуализации
        circle_interval = 10
        result_size = (
            self.image_size
            if np.any(self.image_size < field_dims_arr * circle_interval)
            else field_dims_arr * circle_interval + 50
        )
        response_map = np.zeros(result_size, dtype=np.int16)
        base_img = self._prepare_image()

        # Итерируем по всем точкам тестового поля
        for x in tqdm(range(field_dims_arr[0]), desc=f"  Прогресс для [{cell_type}]", unit="px"):
            for y in range(field_dims_arr[1]):
                coords = (
                    self.image_size[1] // 2 + x - field_dims_arr[0] // 2,
                    self.image_size[0] // 2 + y - field_dims_arr[1] // 2
                )
                # Рисуем точечный стимул
                cv2.circle(base_img, center=coords, radius=1, color=255, thickness=-1) # type: ignore
                resp = cell.get_response(base_img)
                # Удаляем стимул (для корректной работы со следующими стимулами)
                cv2.circle(base_img, center=coords, radius=1, color=0, thickness=-1) # type: ignore
                
                display_coords = (
                    result_size[1] // 2 - (field_dims_arr[0] // 2 - x) * circle_interval,
                    result_size[0] // 2 - (field_dims_arr[1] // 2 - y) * circle_interval
                )
                # Наносим отклик на карту
                cv2.circle(response_map, center=display_coords, radius=2, color=int(resp), thickness=-1) # type: ignore

        return response_map

    def rotate_line_response(
        self,
        cell: ReceptiveField,
        length: int = 100,
        step_angle: int = 10
    ) -> Tuple[List[float], List[int]]:
        """
        Args:
            cell (ReceptiveField): тестируемая клетка
            length (int, optional): длина линии в пикселях
            step_angle (int, optional): шаг угла обзора в градусах
        Returns:
            Tuple[List[float], List[int]]: списки откликов и углов
        """
        img = self._prepare_image()
        h, w = img.shape
        responses = []
        angles = list(range(0, 360, step_angle))
        # Рисуем и анализируем линию под разными углами
        for angle_deg in angles:
            theta = np.deg2rad(angle_deg)
            dx, dy = int(length * np.cos(theta)), int(length * np.sin(theta))
            pt1 = (w // 2 + dx, h // 2 + dy)
            pt2 = (w // 2 - dx, h // 2 - dy)
            # Закрашиваем, чтобы получить отклик
            cv2.line(img, pt1, pt2, color=255, thickness=1) # type: ignore
            responses.append(cell.get_response(img))
            # Отчищаем перед следующей закраской
            cv2.line(img, pt1, pt2, color=0, thickness=1) # type: ignore
        return responses, angles

    def shift_line_response(
        self,
        cell: ReceptiveField,
        max_shift: int = 10
    ) -> Tuple[List[float], List[int]]:
        """
        Args:
            cell (ReceptiveField): тестируемая клетка
            max_shift (int, optional): максимальный сдвиг линии в пикселях
        Returns:
            Tuple[List[float], List[int]]: отклики и значения сдвигов
        """
        img = self._prepare_image()
        h, w = img.shape
        shifts = list(range(-max_shift, max_shift))
        responses = []
        # Сдвигаем вертикальную линию и измеряем отклик
        for shift in shifts:
            x = w // 2 + shift
            # Закрашиваем, чтобы получить отклик
            cv2.line(img, (x, 0), (x, h), color=255, thickness=1) # type: ignore
            responses.append(cell.get_response(img))
            # Отчищаем перед следующей закраской
            cv2.line(img, (x, 0), (x, h), color=0, thickness=1) # type: ignore
        return responses, shifts

    def check_circle_response(
        self,
        cell: ReceptiveField,
        max_radius: int = 30
    ) -> List[float]:
        """
        Args:
            cell (ReceptiveField): тестируемая клетка
            max_radius (int, optional): максимальный радиус круга в пикселях
        Returns:
            List[float]: список откликов для каждого радиуса
        """
        img = self._prepare_image()
        h, w = img.shape
        responses = []
        # Рисуем заполненный круг с меняющимся радиусом
        for radius in range(max_radius):
            cv2.circle(img, (w // 2, h // 2), radius, color=255, thickness=-1) # type: ignore
            responses.append(cell.get_response(img))
            cv2.circle(img, (w // 2, h // 2), radius, color=0, thickness=-1) # type: ignore
        return responses

    def run_all(
        self,
        point_field: Tuple[int, int] = (13, 13),
        length: int = 100,
        step_angle: int = 10,
        max_shift: int = 10,
        max_radius: int = 30,
        cell_types: Optional[CellTypes] = None
    ) -> None:
        """
        Args:
            point_field (Tuple[int, int], optional): размер поля для точечного теста
            length (int, optional): длина линии в пикселях для теста
            step_angle (int, optional): шаг угла обзора в градусах для теста
            max_shift (int, optional): максимальный сдвиг линии в пикселях для теста
            max_radius (int, optional): максимальный радиус круга для теста
            cell_types (Optional[CellTypes]], optional): типы клеток для запуска, например,
                `["simple", "complex"]` только для запуска простых и сложных клеток.
        Returns:
            None
        """
        # Создаем папку для сохранения изображений
        os.makedirs("images", exist_ok=True)
        # Запускаем все тесты для выбранных клеток
        for cell_type, cell in self.cells.items():
            if cell_types is None or cell_type in cell_types:
                # Точечный стимул
                map_img = self.check_point_response(cell_type, cell, point_field)
                plt.title(cell_type)
                plt.imshow(map_img)
                plt.savefig(f"images/{cell_type}_point.png")
                plt.show()
                
                # Нормализованный вариант
                norm_img = cv2.normalize(map_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
                norm_img[norm_img == norm_img[0,0]] = 0
                cv2.imwrite(f"images/{cell_type}_point_norm.png", norm_img)
                
                # Отклик на вращающуюся линию
                res, angles = self.rotate_line_response(cell, length, step_angle)
                plt.title(f"{cell_type} rotate")
                plt.plot(angles, res)
                plt.savefig(f"images/{cell_type}_rotate.png")
                plt.show()
                
                # Отклик на сдвигающуюся линию
                res_s, shifts = self.shift_line_response(cell, max_shift)
                plt.title(f"{cell_type} shift")
                plt.plot(shifts, res_s)
                plt.savefig(f"images/{cell_type}_shift.png")
                plt.show()
                
                # Отклик на круг
                circle_res = self.check_circle_response(cell, max_radius)
                plt.title(f"{cell_type} circle")
                plt.plot(list(range(max_radius)), circle_res)
                plt.savefig(f"images/{cell_type}_circle.png")
                plt.show()


if __name__ == '__main__':
    analyzer = ReceptiveFieldAnalyzer(
        ganglion_cell=GanglionCell(),
        simple_cell=SimpleCell(
            ganglion_layout=(5, 5),
            span=3,
            is_off_type=False,
            orientation=ORIENTATION_VERTICAL
        ),
        complex_cell=ComplexCell(
            simple_layout=(5, 1),
            span=1,
            ganglion_layout=(5, 5),
            ganglion_span=3,
            is_off_type=False,
            orientation=ORIENTATION_VERTICAL
        )
    )
    
    types: CellTypes = ['ganglion', 'simple', 'complex']
    analyzer.run_all(
        point_field=(13, 13),
        length=100,
        step_angle=10,
        max_shift=10,
        max_radius=30,
        cell_types=types
    )