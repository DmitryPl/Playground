from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.boxer.models import Order, Track
from src.boxer.packing import PackingNode


def cuboid_data(o, size=(1, 1, 1)):
    cube = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    cube = np.array(cube).astype(float)
    for i in range(3):
        cube[:, :, i] *= size[i]
    cube += np.array(o)
    return cube


def plot_cube_at(positions, sizes, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data(p, size=s))
    return Poly3DCollection(np.concatenate(g), facecolors=np.repeat(colors, 6, axis=0), **kwargs)


def draw_orders(track: Track, orders: List[Order]) -> None:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    positions = [[order.box.x, order.box.y, order.box.z] for order in orders]
    sizes = [[order.box.length, order.box.width, order.box.height] for order in orders]

    pc = plot_cube_at(positions, sizes, edgecolor="k")
    ax.add_collection3d(pc)

    ax.set_xlim([0, track.length])
    ax.set_ylim([0, track.width])
    ax.set_zlim([0, track.height])

    ax.set_xlabel('length')
    ax.set_ylabel('width')
    ax.set_zlabel('height')

    ax.view_init(30, 30)

    plt.style.use('seaborn-pastel')
    plt.show()


def show_orders(track: Track, orders: List[Order]) -> None:
    for i in range(1, len(orders)):
        draw_orders(track, orders[:i])


def draw_state(node: PackingNode, boxes=True) -> None:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if node.current_solutions and boxes:
        positions = [[order.box.x, order.box.y, order.box.z] for order in node.current_solutions]
        sizes = [[order.box.length, order.box.width, order.box.height] for order in node.current_solutions]
        colors = ['g' if order.fragility else 'b' for order in node.current_solutions]

        pc = plot_cube_at(positions, sizes, colors, edgecolor="k", alpha=0.5)
        ax.add_collection3d(pc)

    xs, ys, zs = [], [], []
    for point in node.extreme_points:
        xs.append(point.x)
        ys.append(point.y)
        zs.append(point.z)
    ax.scatter(xs, ys, zs, c='r', marker='x', alpha=1)

    ax.set_xlim([0, node.track.length])
    ax.set_ylim([0, node.track.width])
    ax.set_zlim([0, node.track.height])

    ax.set_xlabel('x (length)')
    ax.set_ylabel('y (width)')
    ax.set_zlabel('z (height)')

    ax.view_init(30, 30)

    plt.style.use('seaborn-pastel')
    plt.show()
