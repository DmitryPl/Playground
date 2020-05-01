from typing import List

from python.src.structures.graph import Edge
from python.src.structures import Heap
from python.src.structures import Matrix


def prim(weight_matrix: Matrix) -> List[Edge]:
    """ Finding minimum spanning tree """
    length = len(weight_matrix)
    edges: List[Edge] = [Edge(0, 0, 0)] * (length - 1)
    heap = Heap()
    visited: List[bool] = [False] * length

    def add(idx: int):
        """ Add Edges from new node to heap """
        visited[idx] = True
        for idy, price in enumerate(weight_matrix[idx]):
            if price == 0 or visited[idy] or idx == idy:
                continue
            heap.push(Edge(price, idx, idy))

    add(0)
    k = 0
    while k < length - 1:
        was, new_edge = True, None
        while was:
            new_edge = heap.pop()
            was = visited[new_edge.dst]  # check dst node
        edges[k] = new_edge
        add(new_edge.dst)
        k += 1
    return edges
