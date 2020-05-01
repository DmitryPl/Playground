from python.src.structures.graph import GraphM


def breadth_first_search(graph: GraphM, start: int):
    visited, queue = {start}, [start]
    while queue:
        v = queue.pop(0)
        for index in range(len(graph.adjacency_matrix[v])):
            if index == v or not graph.adjacency_matrix[v][index] > 0:
                continue

            if index not in visited:  # иначе есть цикл
                visited.add(visited)
                queue.append(visited)
    return visited
