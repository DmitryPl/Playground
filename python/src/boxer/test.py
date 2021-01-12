from src.boxer.draw_boxes import draw_state
from src.boxer.generator import random_task, random_box
from src.boxer.models import Coord
from src.boxer.packing import PackingNode

# generate
track, orders = random_task(20)
orders.insert(10, random_box(track, order_id=20, fragility=True))

# pre-work
v = sum([order.box.width * order.box.height * order.box.length for order in orders])
w = sum([order.weight for order in orders])
print('v: ', track.width * track.height * track.length >= v)
print('w: ', track.lifting_capacity >= w)

# algorithm
root = PackingNode(0, track, [], orders, [Coord(0., 0., 0.)], None, gamma=3)
best_state = 0
node = root
i = 0

while root.state() != 2:
    if node is None:
        break

    # print(i, node.state(), node.level, len(node.potential_locations))
    # print(node.extreme_points)
    print(i)
    # draw_state(node)
    # from time import sleep
    # sleep(0.5)

    if len(node.free_orders) == 0:
        break

    if not node.check_step():
        if i > best_state:
            best_state = i
            draw_state(node.prev_state)
        i -= 1
        node = node.go_back()
        continue

    node = node.go_next()
    i += 1

if i == 20:
    draw_state(node)

print(max(best_state, i))
