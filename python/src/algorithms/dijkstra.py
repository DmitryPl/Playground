from dataclasses import dataclass
from typing import List, Optional

from python.src.structures.graph import GraphM
from python.src.structures import Heap


@dataclass(order=True)
class Edge:
    price: float
    dst: int


def dijkstra(graph: GraphM, start: int, end: int) -> Optional[float]:
    """ Finding minimum path from "start" node to "end" node """
    heap = Heap()
    heap.push(Edge(0, start))
    visited: List[bool] = [False] * len(graph)
    while not heap.empty():
        min_edge: Edge = heap.pop()
        if visited[min_edge.dst]:
            continue
        visited[min_edge.dst] = True
        if min_edge.dst == end:
            return min_edge.price
        for edge in graph.connections_from(min_edge.dst):
            if visited[edge[0].index]:
                continue
            heap.push(Edge(min_edge.price + edge[1], edge[0].index))
    return None


def test():
    g = GraphM.create_from(['a', 'b', 'c', 'd', 'e', 'f'])
    g.set_connect_asymmetric(0, 1, 5)
    g.set_connect_symmetric(0, 2, 10)
    g.set_connect_asymmetric(0, 4, 2)
    g.set_connect_asymmetric(1, 2, 2)
    g.set_connect_symmetric(1, 3, 4)
    g.set_connect_symmetric(2, 3, 7)
    g.set_connect_symmetric(2, 5, 10)
    g.set_connect_symmetric(3, 4, 3)
    print(dijkstra(g, 0, 5))
