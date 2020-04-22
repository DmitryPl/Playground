from src.structures.graph import GraphM


def depth_first_search(graph: GraphM, start: int):
    visited, stack = {start}, [start]
    while stack:
        v = stack.pop()
        for index in range(len(graph.adjacency_matrix[v])):
            if index == v or not graph.adjacency_matrix[v][index] > 0:
                continue

            if index not in visited:
                visited.add(index)
                stack.append(index)
    return visited
