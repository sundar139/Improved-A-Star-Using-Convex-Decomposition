import cv2
import numpy as np
import random
import math
import time
import os
from skimage.draw import line


class Node:
    """Node class for RRT*"""
    def __init__(self, position, parent=None, cost=0):
        self.position = position
        self.parent = parent
        self.cost = cost

def is_valid_point(point, img):
    """Check if the point is within bounds and not an obstacle."""
    x, y = point
    height, width = img.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return img[y, x] == 255

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_nearest_node(nodes, random_point):
    """Find the nearest node in the tree to the random point."""
    nearest_node = nodes[0]
    min_distance = euclidean_distance(nearest_node.position, random_point)
    for node in nodes:
        distance = euclidean_distance(node.position, random_point)
        if distance < min_distance:
            nearest_node = node
            min_distance = distance
    return nearest_node

def steer(from_point, to_point, max_step_size):
    """Steer from one point to another within a maximum step size."""
    direction = np.array(to_point) - np.array(from_point)
    distance = np.linalg.norm(direction)
    if distance <= max_step_size:
        return tuple(map(int, to_point))
    direction = direction / distance
    new_point = np.array(from_point) + direction * max_step_size
    return tuple(map(int, new_point))

def is_collision_free(img, p1, p2):
    """Check if the path between two points is collision-free."""
    x0, y0 = p1
    x1, y1 = p2
    rr, cc = line(y0, x0, y1, x1)
    for r, c in zip(rr, cc):
        if img[r, c] != 255:
            return False
    return True

def rrt_star(img, start, goal, max_iterations=50000, max_step_size=10, goal_sample_rate=0.1, radius=15):
    """RRT* algorithm implementation."""
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for i in range(max_iterations):
        if random.random() < goal_sample_rate:
            random_point = goal
        else:
            random_point = (random.randint(0, img.shape[1] - 1), random.randint(0, img.shape[0] - 1))

        nearest_node = get_nearest_node(nodes, random_point)
        new_point = steer(nearest_node.position, random_point, max_step_size)

        if is_valid_point(new_point, img) and is_collision_free(img, nearest_node.position, new_point):
            new_node = Node(new_point, parent=nearest_node)

            new_node.cost = nearest_node.cost + euclidean_distance(new_node.position, nearest_node.position)

            near_nodes = [node for node in nodes if euclidean_distance(node.position, new_node.position) <= radius]

            min_cost = new_node.cost
            best_parent = nearest_node
            for near_node in near_nodes:
                if is_collision_free(img, near_node.position, new_node.position):
                    cost = near_node.cost + euclidean_distance(near_node.position, new_node.position)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near_node

            new_node.parent = best_parent
            new_node.cost = min_cost

            for near_node in near_nodes:
                if near_node == best_parent:
                    continue
                if is_collision_free(img, new_node.position, near_node.position):
                    cost = new_node.cost + euclidean_distance(new_node.position, near_node.position)
                    if cost < near_node.cost:
                        near_node.parent = new_node
                        near_node.cost = cost

            nodes.append(new_node)

            if euclidean_distance(new_node.position, goal) <= max_step_size:
                if is_collision_free(img, new_node.position, goal):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + euclidean_distance(new_node.position, goal)
                    nodes.append(goal_node)
                    return goal_node, nodes

    return None, nodes

def reconstruct_path(node):
    """Reconstruct the path from the goal to the start."""
    path = []
    current = node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]

def draw_grid(img, spacing=20):
    """Overlay a grid on the image."""
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for x in range(0, img_color.shape[1], spacing):
        cv2.line(img_color, (x, 0), (x, img_color.shape[0]), (200, 200, 200), 1)
    for y in range(0, img_color.shape[0], spacing):
        cv2.line(img_color, (0, y), (img_color.shape[1], y), (200, 200, 200), 1)

    return img_color

def visualize_tree_and_path(img_original, nodes, path=None, start=None, goal=None):
    """Visualize the RRT* search tree and the final path on the original image."""
    img_color = draw_grid(img_original)

    for node in nodes:
        if node.parent is not None:
            cv2.line(img_color, node.position, node.parent.position, (200, 200, 200), 1)

    if path:
        for i in range(len(path) - 1):
            cv2.line(img_color, path[i], path[i + 1], (0, 255, 0), 2)
        for point in path:
            cv2.circle(img_color, point, 3, (0, 0, 255), -1)

    if start:
        cv2.circle(img_color, start, 5, (0, 255, 0), -1)
    if goal:
        cv2.circle(img_color, goal, 5, (0, 0, 255), -1)

    return img_color

def execute_rrt_star(img_binary, img_original, start, goal):
    """Execute RRT* and visualize results."""
    start_time = time.time()
    goal_node, nodes = rrt_star(img_binary, start, goal)
    end_time = time.time()

    if goal_node:
        path = reconstruct_path(goal_node)

        edge_lengths = [euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1)]
        path_length = sum(edge_lengths)
        variance = np.var(edge_lengths)
        nodes_expanded = len(nodes)
        time_taken = end_time - start_time

        print(f"Path Length: {path_length:.2f}, Variance: {variance:.2f}, "
              f"Nodes Expanded: {nodes_expanded}, Time Taken: {time_taken:.2f}s.")

        img_result = visualize_tree_and_path(img_original, nodes, path, start, goal)
    else:
        nodes_expanded = len(nodes)
        time_taken = end_time - start_time
        print(f"No path found! Nodes Expanded: {nodes_expanded}, Time Taken: {time_taken:.2f}s.")

        img_result = visualize_tree_and_path(img_original, nodes, start=start, goal=goal)

    cv2.imshow("RRT* Pathfinding", img_result)

    while True:
        if cv2.getWindowProperty("RRT* Pathfinding", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(100)
        if key != -1:
            break
    cv2.destroyAllWindows()


def main():
    image_path = "Images/S1.png"
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    img_original = cv2.imread(image_path)
    if img_original is None:
        print("Error: Unable to read the image. Check the path.")
        return

    gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY)

    choice = input("Enter '1' to input coordinates manually or '2' to select on the image: ").strip()
    if choice == '1':
        try:
            start_x = int(input("Enter start x-coordinate: "))
            start_y = int(input("Enter start y-coordinate: "))
            end_x = int(input("Enter end x-coordinate: "))
            end_y = int(input("Enter end y-coordinate: "))
            start = (start_x, start_y)
            goal = (end_x, end_y)

            if not is_valid_point(start, img_binary) or not is_valid_point(goal, img_binary):
                print("Error: Start or goal position is invalid.")
                return

            execute_rrt_star(img_binary, img_original, start, goal)
        except ValueError:
            print("Invalid coordinates. Please enter valid integers.")
    elif choice == '2':
        global start_point, goal_point, rrt_started
        start_point = None
        goal_point = None
        rrt_started = False

        def mouse_callback(event, x, y, flags, param):
            """Handle mouse clicks for selecting points."""
            global start_point, goal_point, rrt_started
            if event == cv2.EVENT_LBUTTONDOWN:
                if start_point is None:
                    start_point = (x, y)
                    print(f"Start point selected at {start_point}")
                elif goal_point is None:
                    goal_point = (x, y)
                    print(f"Goal point selected at {goal_point}")
                    rrt_started = True
                    cv2.setMouseCallback("RRT* Pathfinding", lambda *args: None)
                    execute_rrt_star(img_binary, img_original, start_point, goal_point)

        cv2.namedWindow("RRT* Pathfinding")
        cv2.setMouseCallback("RRT* Pathfinding", mouse_callback)

        while True:
            img_with_grid = draw_grid(img_original.copy())
            if start_point:
                cv2.circle(img_with_grid, start_point, 5, (0, 255, 0), -1)
            if goal_point:
                cv2.circle(img_with_grid, goal_point, 5, (0, 0, 255), -1)

            cv2.imshow("RRT* Pathfinding", img_with_grid)

            if cv2.getWindowProperty("RRT* Pathfinding", cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(100)
            if rrt_started or key == 27:
                break

        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
