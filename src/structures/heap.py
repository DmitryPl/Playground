from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import List, TypeVar, Optional

T = TypeVar('T')


@dataclass
class Heap:
    """ Heap, returning minimum element based on python stdlib
    """
    data: List[T] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    def push(self, a: T) -> None:
        heappush(self.data, a)

    def min(self) -> T:
        return self.data[0]

    def pop(self) -> T:
        return heappop(self.data)

    def empty(self) -> bool:
        return len(self.data) == 0


@dataclass
class MyHeap:
    nodes: List[T] = field(default_factory=list)

    def __init__(self, nodes: List[T]) -> None:
        self.nodes.append(nodes)
        self.min_heapify()

    def __getitem__(self, index: int) -> T:
        return self.nodes[index]

    def __len__(self) -> int:
        return len(self.nodes)

    @staticmethod
    def __parent_index(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def __left_index(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def __right_index(i: int) -> int:
        return 2 * i + 2

    def root(self) -> T:
        return self.nodes[0]

    def parent(self, i: int) -> Optional[T]:
        if 0 < i < len(self):
            return self[(i - 1) // 2]
        return None

    def left(self, i: int) -> Optional[T]:
        if 0 <= i < len(self) and 2 * i + 1 < len(self):
            return self[2 * i + 1]
        return None

    def right(self, i: int) -> Optional[T]:
        if 0 <= i < len(self) and 2 * i + 2 < len(self):
            return self[2 * i + 2]
        return None

    def min(self) -> T:
        return self.nodes[0]

    def min_heapify_subtree(self, i: int) -> None:
        """ Heapify at a node assuming all subtrees are heapified """
        size = len(self)
        left = self.__left_index(i)
        right = self.__right_index(i)
        minimum = i
        if left < size and self.nodes[left] < self.nodes[minimum]:
            minimum = left
        if right < size and self.nodes[right] < self.nodes[minimum]:
            minimum = right
        if minimum != i:
            self.nodes[i], self.nodes[minimum] = self.nodes[minimum], self.nodes[i]
            self.min_heapify_subtree(minimum)

    def pop(self) -> Optional[T]:
        minimum = self.nodes[0]
        if len(self) > 1:
            self.nodes[0] = self.nodes[-1]
            self.nodes.pop()
            self.min_heapify_subtree(0)
        elif len(self) == 1:
            self.nodes.pop()
        else:
            return None
        return minimum

    def decrease_key(self, i: int, val: T) -> None:
        """ Update node value, bubble it up as necessary to maintain heap property """
        self.nodes[i] = val
        parent = self.parent(i)
        while i != 0 and self.nodes[parent] > self.nodes[i]:
            self.nodes[parent], self.nodes[i] = self.nodes[i], self.nodes[parent]
            i = parent
            parent = self.parent(i) if i > 0 else None

    def min_heapify(self) -> None:
        """ Heapify an un-heapified array """
        for i in range(len(self.nodes), -1, -1):
            self.min_heapify_subtree(i)
