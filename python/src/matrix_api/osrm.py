"""Модуль с утилитами для обращения к OSRM серверу."""
import urllib
from random import uniform
from typing import Optional, Tuple, Union
from urllib.parse import quote

import numpy as np
import requests
from loguru import logger
from polyline import encode as polyline_encode

Array = Union[np.ndarray, list[tuple[float, float]]]


def get_osrm_matrix(
        src: Array,
        dst: Optional[Array] = None,
        *,
        transport: str = "car",
        profile: str = "driving",
        return_distances: bool = True,
        return_durations: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Получаем матрицу расстояний.
    Матрица от src до dst, или от src до src, если dst не указан.
    Эта функция является "фронтендом", она обрабатывает параметры, обеспечивает удобный интерфейс,
    вызывает исправление решения и так далее.
    Всю остальную работу эта функция передает функции table.
    Parameters
    ----------
    src : Набор точек, из которых мы считаем расстояния
    dst : Набор точек до куда считаем расстояния. Если None, то считаем (src, src)
    transport : {'car', 'foot', 'bicycle'} Тип транспорта, профайл
    profile : Не знаю, что это. Но osrm это принимает.
    return_distances : Возвращать ли расстояния
    return_durations : Возвращать ли время
    Returns
    -------
    Возвращаемся матрицу расстояния и/или времени
    """
    assert len(src) != 0, "Не заданы начальные точки"
    assert dst is None or len(dst) != 0, "Не заданы конечные точки"
    assert return_distances or return_durations, "Ничего не возвращаем"

    src = np.array(src, dtype=np.float32)
    dst = dst if dst is None else np.array(dst, dtype=np.float32)

    host = dict(  # выбираем сервер, к которому обращаемся
        car="http://api.severtrans.moskovskiy.org:5000",
        foot="http://api.severtrans.moskovskiy.org:5000",
        bicycle="http://api.severtrans.moskovskiy.org:5000",
    )[transport]

    distances, durations = _table(
        host=host,
        src=src,
        dst=dst,
        profile=profile,
        return_distances=return_distances,
        return_durations=return_durations,
    )

    assert (durations is None or np.abs(durations).sum() != 0) and (  # проверяем, что матрица не нулевая
            distances is None or np.abs(distances).sum() != 0
    ), f"OSRM вернул 0 матрицу. Проверьте порядок координат." f"{src}" f"{durations}" f"{distances}"

    return distances, durations


def _encode_src_dst(
        src: Array,
        dst: Optional[Array] = None,
        *,
        return_distances: bool = True,
        return_durations: bool = True,
) -> Tuple[str, str]:
    """Кодируем координаты src, dst в виде параметров.
    Parameters
    ----------
    src : Набор точек источников
    dst : Набор точек финишей
    return_distances : Возвращать ли расстояния
    return_durations : Возвращаться ли время
    Returns
    -------
    Закодированный polyline и закодированные params для подстановки в url
    """

    coords = src if dst is None else np.vstack([src, dst])
    polyline = polyline_encode(coords)

    params = {"annotations": ",".join(["duration"] * return_durations + ["distance"] * return_distances)}
    if dst is not None:
        ls, ld = map(len, (src, dst))
        params["sources"] = ";".join(map(str, range(ls)))
        params["destinations"] = ";".join(map(str, range(ls, ls + ld)))

    return quote(polyline), urllib.parse.urlencode(params)


def _table(
        host: str,
        src: Array,
        dst: Optional[Array] = None,
        profile: str = "driving",
        return_distances: bool = False,  # что мы хотим получить в результате
        return_durations: bool = True,
) -> Tuple[Array, Array]:
    """Отправляем запрос матрицы расстояний в OSRM и получаем ответ.
    Матрица с одинаковыми параметрами кешируется при первом вызове.
    Parameters
    ----------
    host : Инстанс OSRM, который мы проверяем
    src : np.array точек, от которых мы считаем расстояние
    dst : np.array точек, до которых мы считаем расстояние
    profile : Какой-то параметр osrm
    return_distances : Возвращать ли расстояния
    return_durations : Возвращать ли время перемещения
    """
    polyline, params = _encode_src_dst(src, dst, return_distances=return_distances, return_durations=return_durations)
    url = f"{host}/table/v1/{profile}/polyline({polyline})?{params}"

    r = None
    try:  # делаем запрос с обработкой ошибок
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:
        logger.info(f"Проблема: {r.content}")
        raise e

    logger.info(f"{r.elapsed} затрачено на вычисление матрицы.")
    parsed_json = r.json()
    di, du = parsed_json.get("distances"), parsed_json.get("durations")

    return (
        np.array(di, dtype=np.int32).tolist() if di is not None else None,
        np.array(du, dtype=np.int32).tolist() if du is not None else None,
    )


def generate_points(n: int, min_x=55.65, max_x=55.82, min_y=37.45, max_y=37.75):
    return [(uniform(min_x, max_x), uniform(min_y, max_y)) for _ in range(n)]


print(get_osrm_matrix(generate_points(20)))
