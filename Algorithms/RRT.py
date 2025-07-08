import cv2
import numpy as np
import random
import math
import time
import os

class Node:
    """Node class for RRT"""
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

def is_valid_point(point, obstacle_map):
    """Check if the point is within bounds and not an obstacle."""
    x, y = point
    height, width = obstacle_map.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return obstacle_map[y, x] == 0

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

def rrt(obstacle_map, start, goal, max_iterations=50000, max_step_size=10, goal_sample_rate=0.1):
    """RRT algorithm implementation."""
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for i in range(max_iterations):
        if random.random() < goal_sample_rate:
            random_point = goal
        else:
            random_point = (random.randint(0, obstacle_map.shape[1] - 1),
                            random.randint(0, obstacle_map.shape[0] - 1))

        nearest_node = get_nearest_node(nodes, random_point)
        new_point = steer(nearest_node.position, random_point, max_step_size)

        if is_valid_point(new_point, obstacle_map):
            new_node = Node(new_point, parent=nearest_node)
            nodes.append(new_node)

            if euclidean_distance(new_node.position, goal) <= max_step_size:
                goal_node.parent = new_node
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


def visualize_tree_and_path(img, nodes, path=None, start=None, goal=None):
    """Visualize the RRT search tree and the final path."""
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

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

def execute_rrt(obstacle_map, img_with_grid, start, goal):
    """Execute RRT and display results."""
    start_time = time.time()
    goal_node, nodes = rrt(obstacle_map, start, goal)
    end_time = time.time()

    if goal_node:
        path = reconstruct_path(goal_node)
        img_with_path = visualize_tree_and_path(img_with_grid, nodes, path, start, goal)
        
        edge_lengths = [euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1)]
        path_length = sum(edge_lengths)
        variance = np.var(edge_lengths)
        
        cv2.imshow("RRT Visualization", img_with_path)
        print(f"Path Length: {path_length:.2f}, Variance: {variance:.2f}, "
              f"Nodes Expanded: {len(nodes)}, Time Taken: {end_time - start_time:.2f}s")
    else:
        print(f"No path found! Nodes Expanded: {len(nodes)}, Time Taken: {end_time - start_time:.2f}s")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img_path = "Images/S1.png"

    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image. Check the path.")
        return

    white_threshold = 200
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, obstacle_map = cv2.threshold(gray_img, white_threshold, 255, cv2.THRESH_BINARY_INV)

    img_with_grid = draw_grid(img.copy())

    choice = input("Enter '1' to input coordinates manually or '2' to select on the image: ").strip()
    if choice == '1':
        try:
            start_x = int(input("Enter start x-coordinate: "))
            start_y = int(input("Enter start y-coordinate: "))
            end_x = int(input("Enter end x-coordinate: "))
            end_y = int(input("Enter end y-coordinate: "))
            start = (start_x, start_y)
            goal = (end_x, end_y)

            if not is_valid_point(start, obstacle_map) or not is_valid_point(goal, obstacle_map):
                print("Error: Start or goal point is invalid.")
                return

            execute_rrt(obstacle_map, img_with_grid, start, goal)
        except ValueError:
            print("Invalid coordinates. Please enter valid integers.")
    elif choice == '2':
        points = []
        path_found = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal points, path_found
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                color = (0, 255, 0) if len(points) == 1 else (0, 0, 255)
                cv2.circle(img_with_grid, (x, y), 5, color, -1)
                cv2.imshow("RRT Visualization", img_with_grid)

                if len(points) == 2:
                    start, goal = points

                    if not is_valid_point(start, obstacle_map) or not is_valid_point(goal, obstacle_map):
                        print("Error: Start or goal point is invalid.")
                        points.clear()
                        return

                    execute_rrt(obstacle_map, img_with_grid, start, goal)
                    path_found = True

        cv2.imshow("RRT Visualization", img_with_grid)
        cv2.setMouseCallback("RRT Visualization", mouse_callback)

        while not path_found:
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
