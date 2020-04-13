from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TypeVar, Optional, Tuple

from src.structures.matrix import Matrix

T = TypeVar('T')


@dataclass
class Node:
    data: T
    index: int


@dataclass(order=True)
class Edge:
    price: float
    src: int
    dst: int

    def __str__(self) -> str:
        return f'{self.src}->{self.dst}'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class GraphL:
    """ Graph over list adjacency """
    nodes: int
    edges: List[Tuple[int, int, float]] = field(default_factory=list)

    def __len__(self) -> int:
        return self.nodes

    def __str__(self) -> str:
        return f'{self.edges}'

    def __repr__(self):
        return str(self)

    def add_edge(self, u: int, v: int, w: float) -> None:
        """ Add edge """
        self.edges.append((u, v, w))


@dataclass
class GraphM:
    """ Graph over matrix adjacency """
    nodes: List[Node]
    adjacency_matrix: Matrix = field(init=False)

    def __post_init__(self):
        self.adjacency_matrix = Matrix(len(self.nodes))

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> str:
        return f'{self.nodes}\n{self.adjacency_matrix}'

    def __repr__(self):
        return str(self)

    @staticmethod
    def create_from(nodes: List[T]) -> GraphM:
        """ Creates graph from list of something """
        temp = [Node(0, 0)] * len(nodes)
        for idx, data in enumerate(nodes):
            temp[idx] = Node(data, idx)
        return GraphM(temp)

    def index_of(self, data: T) -> Optional[Node]:
        """ Get node by data """
        for node in self.nodes:
            if data == node.data:
                return node
        return None

    def set_connect_asymmetric(self, first_node: int, second_node: int, weight: int = 1) -> None:
        """ Connect nodes asymmetric a -> b != b <- a """
        self.adjacency_matrix[first_node][second_node] = weight

    def set_connect_symmetric(self, first_node: int, second_node, weight: int = 1) -> None:
        """ Connect nodes symmetric a -> b == b <- a """
        self.adjacency_matrix[first_node][second_node] = self.adjacency_matrix[second_node][first_node] = weight

    def connections_from(self, node: int) -> List[Tuple[Node, float]]:
        """ Get connections from node """
        return [(self.nodes[col_num], self.adjacency_matrix[node][col_num])
                for col_num in range(len(self.adjacency_matrix[node]))
                if self.adjacency_matrix[node][col_num] != 0]

    def connections_to(self, node: int) -> List[Tuple[Node, float]]:
        """ Get connections to node """
        column = [row[node] for row in self.adjacency_matrix]
        return [(self.nodes[row_num], column[row_num]) for row_num in range(len(column)) if column[row_num] != 0]

    def has_path(self, first_node: int, second_node: int) -> bool:
        """ Path first -> second """
        return self.adjacency_matrix[first_node][second_node] != 0

    def has_conn(self, first_node: int, second_node: int) -> bool:
        """ Some path first <-> second """
        return self.has_path(first_node, second_node) or self.has_path(second_node, first_node)

    def get_weight(self, first_node: int, second_node: int) -> float:
        """ Weight from first -> second """
        return self.adjacency_matrix[first_node][second_node]
