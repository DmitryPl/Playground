from random import randint
from typing import Tuple, List

from src.boxer.models import Track, Order, Box


def small_track() -> Track:
    """ Gazel
    @return: value capacity m3, lifting capacity kg, width-height-length (m)
    """
    return Track(13.5, 2000., 1.860, 1.927, 3.145)


def random_box(track: Track, order_id=0, fragility=False) -> Order:
    """
    @param fragility:
    @param order_id:
    @param track:
    @return: weight, width-height-length
    """

    def rand_size(size) -> float:
        return randint(10, int(size * 1000)) / 1000  # m -> mm -> m

    width, length, height = track.width / 2, track.length / 2, track.height / 2
    box_width, box_height, box_length = rand_size(width), rand_size(height), rand_size(length)
    box_value = box_width * box_height * box_length  # m3
    box_density = randint(500, 1200)  # kg/m3
    box_weight = box_density * box_value  # kg

    return Order(order_id, Box(0, 0, 0, box_width, box_height, box_length), box_weight, box_value, fragility, False)


def random_task(n: int) -> Tuple[Track, List[Order]]:
    track = small_track()
    return track, [random_box(track, order_id=i) for i in range(n)]
