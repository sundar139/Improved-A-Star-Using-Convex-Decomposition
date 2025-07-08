import cv2
import numpy as np
import random
import math
import time
import os
from scipy.spatial import KDTree
from queue import PriorityQueue
from skimage.draw import line


def is_valid_point(point, img):
    """Check if the point is within bounds and not an obstacle."""
    x, y = point
    height, width = img.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return img[y, x] == 255

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_valid_path(img, p1, p2):
    """Check if a direct path between two points is valid."""
    x0, y0 = p1
    x1, y1 = p2
    rr, cc = line(y0, x0, y1, x1)
    for r, c in zip(rr, cc):
        if not is_valid_point((c, r), img):
            return False
    return True

def build_prm(img, num_samples, k, start, goal):
    """Build a PRM graph, ensuring start and goal are included."""
    height, width = img.shape
    samples = []

    while len(samples) < num_samples:
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        if is_valid_point((x, y), img):
            samples.append((x, y))

    if start not in samples:
        samples.append(start)
    if goal not in samples:
        samples.append(goal)

    tree = KDTree(samples)
    graph = {sample: [] for sample in samples}

    for sample in samples:
        distances, indices = tree.query(sample, k + 1)
        for idx in indices[1:]:
            neighbor = samples[idx]
            if is_valid_path(img, sample, neighbor):
                graph[sample].append(neighbor)

    return graph, samples

def find_path(graph, start, goal):
    """Find the shortest path using A* search."""
    if start not in graph or goal not in graph:
        raise ValueError("Start or goal is not in the graph!")

    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    nodes_expanded = 0

    while not pq.empty():
        _, current = pq.get()
        nodes_expanded += 1

        if current == goal:
            break

        for neighbor in graph[current]:
            new_cost = cost_so_far[current] + euclidean_distance(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean_distance(neighbor, goal)
                pq.put((priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return None, nodes_expanded

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]

    return path[::-1], nodes_expanded

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

def visualize_prm(img_original, graph, path=None, start=None, goal=None):
    """Visualize the PRM and the path on the original image."""
    img_color = draw_grid(img_original)

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            cv2.line(img_color, node, neighbor, (200, 200, 200), 1)

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

def execute_prm(img_binary, img_original, start, goal):
    """Execute the PRM algorithm and visualize results."""
    start_time = time.time()

    graph, samples = build_prm(img_binary, 1000, 15, start, goal)
    path, nodes_expanded = find_path(graph, start, goal)

    end_time = time.time()

    if path:
        edge_lengths = [euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1)]
        path_length = sum(edge_lengths)
        variance = np.var(edge_lengths)

        print(f"Path Length: {path_length:.2f}, Variance: {variance:.2f}, "
              f"Nodes Expanded: {nodes_expanded}, Time Taken: {end_time - start_time:.2f}s.")
    else:
        print(f"Pathfinding failed! Nodes Expanded: {nodes_expanded}, "
              f"Time Taken: {end_time - start_time:.2f}s.")

    img_result = visualize_prm(img_original, graph, path, start, goal)
    cv2.imshow("PRM Pathfinding", img_result)

    while True:
        if cv2.getWindowProperty("PRM Pathfinding", cv2.WND_PROP_VISIBLE) < 1:
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

            execute_prm(img_binary, img_original, start, goal)
        except ValueError:
            print("Invalid coordinates. Please enter valid integers.")
    elif choice == '2':
        global start_point, goal_point
        start_point = None
        goal_point = None

        def mouse_callback(event, x, y, flags, param):
            global start_point, goal_point
            if event == cv2.EVENT_LBUTTONDOWN:
                if start_point is None:
                    start_point = (x, y)
                    print(f"Start point selected at {start_point}")
                elif goal_point is None:
                    goal_point = (x, y)
                    print(f"Goal point selected at {goal_point}")
                    execute_prm(img_binary, img_original, start_point, goal_point)

        cv2.namedWindow("PRM Pathfinding")
        cv2.setMouseCallback("PRM Pathfinding", mouse_callback)

        while True:
            img_with_grid = draw_grid(img_original.copy())
            if start_point:
                cv2.circle(img_with_grid, start_point, 5, (0, 255, 0), -1)
            if goal_point:
                cv2.circle(img_with_grid, goal_point, 5, (0, 0, 255), -1)

            cv2.imshow("PRM Pathfinding", img_with_grid)

            if cv2.getWindowProperty("PRM Pathfinding", cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(100)
            if key == 27:
                break

        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
