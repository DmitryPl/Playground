from dataclasses import dataclass
from typing import List


@dataclass
class Box:
    x: float
    y: float
    z: float
    width: float  # y
    height: float  # z
    length: float  # x

    def __str__(self) -> str:
        return f'x: {self.x}, y: {self.y}, z: {self.z}, ' \
               f'l: {self.length}, w: {self.width}, h: {self.height}'


@dataclass
class Track:
    value_capacity: float
    lifting_capacity: float
    width: float
    height: float
    length: float


@dataclass
class Order:
    order_id: int
    box: Box
    weight: float
    value: float
    fragility: bool
    vertical: bool

    @classmethod
    def new_order(cls, length: float, width: float, height: float) -> 'Order':
        order = Order(cls.order_id, cls.box, cls.weight, cls.value, cls.fragility, cls.vertical)
        order.box.length, order.box.width, order.box.height = length, width, height
        return order


Orders = List[Order]


@dataclass
class Coord:
    x: float
    y: float
    z: float

    def __lt__(self, other: 'Coord'):
        """back-left-down"""
        if self.x < other.x:
            return True
        if self.y < other.y:
            return True
        if self.z < other.z:
            return True
        return False


Coords = List[Coord]
