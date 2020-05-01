from typing import List, Optional

from python.src.structures.graph import GraphL


def bellman_ford(graph: GraphL, start: int) -> Optional[List[float]]:
    length = len(graph)
    distance = [float('inf')] * length
    distance[start] = 0.0

    for i in range(length - 1):
        for u, v, w in graph.edges:
            if distance[u] != float('inf') and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w

        for u, v, w in graph.edges:
            if distance[u] != float("Inf") and distance[u] + w < distance[v]:
                print('Graph contains negative weight cycle')
                return None

    return distance


def test():
    g = GraphL(5)
    g.add_edge(0, 1, -1)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 2, 3)
    g.add_edge(1, 3, 2)
    g.add_edge(1, 4, 2)
    g.add_edge(3, 2, 5)
    g.add_edge(3, 1, 1)
    g.add_edge(4, 3, -3)
    print(bellman_ford(g, 0))
