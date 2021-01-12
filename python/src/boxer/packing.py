from copy import copy
from typing import List, Optional, Tuple

from src.boxer.models import Box, Track, Orders, Coords, Coord, Order


class PackingNode:
    # TODO: лучше сделать Node внутри Tree
    # TODO: mix boxes for one point (id)

    def __init__(
            self,
            level: int,
            track: Track,
            current_solutions: Orders,
            free_orders: Orders,
            extreme_points: Coords,
            prev_state: Optional['PackingNode'],
            alpha: float = 0.8,  # поддерживающая площадь
            betta: float = 0.5,  # длина достижимости
            gamma: int = 0,  # кол-во попыток на груз
    ):
        self.level = level
        self._current_state: int = 0
        self.track: Track = track
        self.current_solutions: Orders = current_solutions
        self.free_orders: Orders = free_orders
        self.potential_locations: List[Tuple[Coords, Order]] = []
        self.extreme_points: Coords = extreme_points
        self.next_state: Optional['PackingNode'] = None
        self.prev_state: Optional['PackingNode'] = prev_state

        self.alpha: float = alpha
        self.betta: float = betta
        self.gamma: int = gamma

        if not self._check_init(alpha, betta, gamma, free_orders):
            return
        self._get_locations()

    def __str__(self) -> str:
        return f'\n' \
               f'current state: {self._current_state}\n' \
               f'track: l:{self.track.length} w:{self.track.width} h:{self.track.height}\n' \
               f'current: {len(self.current_solutions)},  free: {len(self.free_orders)}\n' \
               f'potential: {len(self.potential_locations)}\n' \
               f'extreme points: {len(self.extreme_points)}' \
               f'\n'

    def state(self) -> int:
        return self._current_state

    def check_step(self) -> bool:
        if self.gamma != 0 and self._current_state + 1 == self.gamma:  # или мы можем перебрать все или не все
            return False
        if not self.potential_locations:  # там вообще что-то есть
            return False
        if self._current_state == len(self.potential_locations):  # еще остались состояния
            return False
        return True

    def go_next(self) -> Optional['PackingNode']:
        if self.gamma != 0 and self._current_state + 1 == self.gamma:  # или мы можем перебрать все или не все
            return None
        if not self.potential_locations:  # там вообще что-то есть
            return None
        if self._current_state == len(self.potential_locations):  # еще остались состояния
            return None

        new_extreme_points, new_state = self.potential_locations[self._current_state]
        new_current_solution = copy(self.current_solutions) + [new_state]
        new_extreme_point = self._new_extreme_points(new_extreme_points, new_state.box)

        new_node = PackingNode(
            level=self.level + 1,
            track=self.track,
            current_solutions=new_current_solution,
            free_orders=copy(self.free_orders),
            extreme_points=new_extreme_point,
            alpha=self.alpha,
            betta=self.betta,
            gamma=self.gamma,
            prev_state=self
        )
        self._current_state += 1
        return new_node

    def go_back(self) -> Optional['PackingNode']:
        return self.prev_state

    def _new_extreme_points(self, extreme_points: Coords, box: Box) -> Coords:
        ep_set = set()
        min_z, max_z = box.z, box.z + box.height
        min_y, max_y = box.y, box.y + box.width
        min_x, max_x = box.x, box.x + box.length

        for point in self.extreme_points:
            # за коробкой, не включая верхнуюю и правую грань области
            if 0 <= point.x <= box.x and min_y <= point.y < max_y and min_z <= point.z < max_z:
                continue
            # на левой грани, не включая верхнее и ближнее ребро
            if point.y == box.y and min_x <= point.x < max_x and min_z <= point.z < max_z:
                continue
            # на нижней грани, не включая ближнее и правое ребро
            if point.z == box.z and min_x <= point.x < max_x and min_y <= point.y < max_y:
                continue

            ep_set.add((point.x, point.y, point.z))

        ep_set.update([(point.x, point.y, point.z) for point in extreme_points])
        return [Coord(point[0], point[1], point[2]) for point in ep_set]

    def _get_locations(self):
        order = self.free_orders.pop()
        x, y, z = order.box.length, order.box.width, order.box.height

        if order.vertical:
            states = [(x, y, z), (y, x, z), (z, y, x), (x, z, y), (y, z, x), (z, x, y)]  # в теории можно сортировать
        else:
            states = [(x, y, z), (y, x, z)]

        if order.fragility:
            self.extreme_points = self._sort_back_left_up(self.extreme_points)
        else:
            self.extreme_points = self._sort_back_left_down(self.extreme_points)

        for point in self.extreme_points:  # проверяем + сразу вычисляем новые точки
            self._check_extreme_point(order, states, point)

    def _check_extreme_point(self, order: Order, states: List[Tuple], point: Coord):
        for state in states:
            length, width, height = state
            tmp_box = Box(point.x, point.y, point.z, width, height, length)
            coord = self._check_potential_solution(tmp_box, order.fragility)
            if coord is None:
                continue
            tmp_order = copy(order)
            tmp_order.box = tmp_box
            self.potential_locations.append((coord, tmp_order))

    def _check_potential_solution(self, box: Box, fragility: bool) -> Optional[Coords]:
        """
        Мы должны проверить:
        - коробка может там быть, стоять, существовать, реально достать
        - 6 дополнительных экстремальных точек
        @param box:
        @return:
        """
        if not self._in_track(self.track, box):  # если не в пределах объема
            # print('out', box.x, box.y, box.z)
            return None

        box_square = box.width * box.length  # плозадь пов-ти xy
        support_area = 0.  # поддерживающая площадь под коробкой
        max_x = 0  # ближайшая к нам сторона других коробок по проекции от (x, y, z)
        points = [
            Coord(0., box.y, box.z + box.height),  # 0 x
            Coord(box.x, 0., box.z + box.height),  # 1 y
            Coord(0., box.y + box.width, box.z),  # 2 x
            Coord(box.x, box.y + box.width, 0.),  # 3 z
            Coord(box.x + box.length, box.y, 0.),  # 4 z
            Coord(box.x + box.length, 0., box.z),  # 5 y
        ]  # проекции 3 точек на плоскости (x, y, z + h), (x, y + w, z), (x + l, y, z)

        if not fragility:
            points.append(Coord(box.x, box.y, box.z + box.height))
            if box.length > self.betta:  # дополнительная точка, если коробка длинная
                points.append(Coord(box.x + box.length - self.betta, box.y, box.z + box.height))

        for solution in self.current_solutions:
            if self._is_overlapping_3d(box, solution.box):  # с кем-то пересекается
                # print('overlapping', box.x, box.y, box.z)
                return None

            max_x = self._get_max_x(max_x, solution.box, box)
            if box.z != 0:
                area = self._get_support_area(box, solution.box)
            else:
                area = -1

            if area != -1:
                support_area += area
            if area != -1 and solution.fragility:  # заказ под ним не хрупкий
                # print('fragility', box.x, box.y, box.z)
                return None
            if max_x - (box.x + box.length) > self.betta:  # не попали в длину достижимости
                # print('too far', box.x, box.y, box.z)
                return None

            if not self._check_front_area(solution.box, box):  # перекрывает доступ для укладки
                # print('bad front area')
                return None

            self._update_extreme_points(points, solution.box, box)  # обновляем экстремальные точки

        if box.z != 0 and support_area / box_square < self.alpha:  # не попали в минимальную поддерживающую площадь
            # print('area', box.x, box.y, box.z, 'area', support_area / box_square)
            return None

        return points

    @staticmethod
    def _check_init(alpha: float, betta: float, gamma: float, free_orders: Orders) -> bool:
        if not 0. <= alpha <= 1:
            return False
        if not betta >= 0:
            return False
        if not gamma >= 0:
            return False
        if not len(free_orders) > 0:
            return False
        return True

    @staticmethod
    def _update_extreme_points(points: Coords, a: Box, b: Box):
        """ Обновляем экстремальные точки
        @param points: 6 штук, подробнее далее
        @param a: об кого обновляемся (solution box)
        @param b: из-за чего обновляемся (current box)
        @return: ничего, там пусть и обновляет
        """

        # 0, 1: (x, y, z + h) проекция на: zy (x++; (x - ?, y, z + h)), zx (y++; (x, y - ?, z + h))
        # 2, 3: (x, y + w, z) проекция на: zy (x++; (x - ?, y + w, z)), xy (z++; (x, y + w, z - ?))
        # 4, 5: (x + l, y, z) проекция на: xy (z++; (x + l, y, z - ?)), xz (y++; (x + l, y - ?, z))

        max_x, min_x = a.x + a.length, a.x
        max_y, min_y = a.y + a.width, a.y
        max_z, min_z = a.z + a.height, a.z

        if 0 <= min_x <= b.x:  # yz
            if min_y <= b.y <= max_y and min_z <= b.z + b.height <= max_z:  # 0
                x = min(max_x, b.x)
                points[0].x = max(points[0].x, x)
            if min_y <= b.y + b.width <= max_y and min_z <= b.z <= max_z:  # 2
                x = min(max_x, b.x)
                points[2].x = max(points[2].x, x)

        if 0 <= min_y <= b.y:  # xz
            if min_x <= b.x <= max_x and min_z <= b.z + b.height <= max_z:  # 1
                y = min(max_y, b.y)
                points[1].y = max(points[1].y, y)
            if min_x <= b.x + b.length <= max_x and min_z <= b.z <= max_z:  # 5
                y = min(max_y, b.y)
                points[5].y = max(points[5].y, y)

        if 0 <= min_z <= b.z:  # xy
            if min_x <= b.x <= max_x and min_y <= b.y + b.width <= max_y:  # 3
                z = min(max_z, b.z)
                points[3].z = max(points[3].z, z)
            if min_x <= b.x + b.length <= max_x and min_y <= b.y <= max_y:  # 4
                z = min(max_z, b.z)
                points[4].z = max(points[4].z, z)

    @staticmethod
    def _in_track(track: Track, box: Box) -> bool:
        """check that box in track"""
        x = box.x + box.length < track.length
        y = box.y + box.width < track.width
        z = box.z + box.height < track.height
        return x and y and z

    @staticmethod
    def _is_overlapping_1d(min_a, max_a, min_b, max_b) -> bool:
        return max_a > min_b and max_b > min_a

    @staticmethod
    def _is_overlapping_3d(a: Box, b: Box) -> bool:
        """ перекрываются ли два ящика, касание устраивает
        @param a:
        @param b:
        @return: пересекаются или нет
        """

        x = PackingNode._is_overlapping_1d(a.x, a.x + a.length, b.x, b.x + b.length)
        y = PackingNode._is_overlapping_1d(a.y, a.y + a.width, b.y, b.y + b.width)
        z = PackingNode._is_overlapping_1d(a.z, a.z + a.height, b.z, b.z + b.height)
        return x and y and z

    @staticmethod
    def _get_support_area(a: Box, b: Box) -> float:
        """ support area of box a for box b
        @param a: under the b?
        @param b: on the a?
        @return: square
        """
        level_a = a.z
        level_b = b.z + b.height

        if level_b != level_a:
            return 0

        x_overlap = max(0., min(a.x + a.length, b.x + b.length) - max(a.x, b.x))
        y_overlap = max(0., min(a.y + a.width, b.y + b.width) - max(a.y, b.y))
        overlap_area = x_overlap * y_overlap
        return overlap_area

    @staticmethod
    def _check_front_area(a: Box, b: Box) -> bool:
        """ a не перекрывает доступ к b по оси ZY
        @param a:
        @param b:
        @return:
        """
        if a.x <= b.x + b.length:
            return True

        y = PackingNode._is_overlapping_1d(a.y, a.y + a.width, b.y, b.y + b.width)
        z = PackingNode._is_overlapping_1d(a.z, a.z + a.height, b.z, b.z + b.height)
        return y and z

    @staticmethod
    def _sort_back_left_down(corners: Coords) -> Coords:
        return sorted(corners, key=lambda coord: (coord.x, coord.y, coord.z))

    @staticmethod
    def _sort_back_left_up(corners: Coords) -> Coords:
        return sorted(corners, key=lambda coord: (coord.x, coord.y, -coord.z))

    @staticmethod
    def _get_max_x(max_x: float, a: Box, b: Box) -> float:
        """ ищем самую длинную проекцию, по ближайшим стенкам по оси их, для проверки доставаемости
        @param max_x: текущая проекция
        @param a: к ней проекция
        @param b: от нее проекция
        @return: длина
        """
        min_b, max_b = b.y, b.y + b.width
        min_a, max_a = a.y, a.y + a.width

        if PackingNode._is_overlapping_1d(min_a, max_a, min_b, max_b):
            return max(a.x + a.length, max_x)
        return max_x


class PackingTree:
    def __init__(self, track: Track, orders: Orders, alpha: float, betta: float, gamma: int):
        self.alpha = alpha
        self.betta = betta
        self.gamma = gamma
        self.track = track

        current_solution = []
        free_orders = orders

        self.root = PackingNode(
            level=0,
            track=track,
            current_solutions=current_solution,
            free_orders=free_orders,
            extreme_points=[],
            alpha=alpha,
            betta=betta,
            gamma=gamma,
            prev_state=None
        )
