import cv2
import numpy as np
import math
import heapq
import time
import statistics

class Node:
    """A node class for A* Pathfinding"""
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h 
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_valid_point(x, y, obstacle_map):
    """Check if the point is within image bounds and not an obstacle"""
    height, width = obstacle_map.shape
    return 0 <= x < width and 0 <= y < height and obstacle_map[y, x] == 0

def get_neighbors(current, obstacle_map):
    """Get valid neighboring points"""
    neighbors = [
        (current[0] + 1, current[1]),
        (current[0] - 1, current[1]),
        (current[0], current[1] + 1),
        (current[0], current[1] - 1),
        (current[0] + 1, current[1] + 1),
        (current[0] - 1, current[1] - 1),
        (current[0] + 1, current[1] - 1),
        (current[0] - 1, current[1] + 1)
    ]
    return [neighbor for neighbor in neighbors if is_valid_point(neighbor[0], neighbor[1], obstacle_map)]

def astar(obstacle_map, start, end):
    """A* pathfinding algorithm"""
    start_node = Node(start, g=0, h=heuristic(start, end))
    open_list = []
    open_dict = {start: start_node}
    closed_set = set()
    heapq.heappush(open_list, start_node)
    nodes_expanded = 0

    while open_list:
        current_node = heapq.heappop(open_list)
        del open_dict[current_node.position]
        closed_set.add(current_node.position)
        nodes_expanded += 1

        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], nodes_expanded

        neighbors = get_neighbors(current_node.position, obstacle_map)
        for neighbor_pos in neighbors:
            if neighbor_pos in closed_set:
                continue

            diagonal = (abs(neighbor_pos[0] - current_node.position[0]) +
                        abs(neighbor_pos[1] - current_node.position[1])) == 2
            move_cost = 1.414 if diagonal else 1
            g_cost = current_node.g + move_cost
            h_cost = heuristic(neighbor_pos, end)

            if neighbor_pos in open_dict:
                existing_node = open_dict[neighbor_pos]
                if g_cost < existing_node.g:
                    existing_node.g = g_cost
                    existing_node.f = g_cost + h_cost
                    existing_node.parent = current_node
            else:
                neighbor_node = Node(neighbor_pos, g=g_cost, h=h_cost)
                neighbor_node.parent = current_node
                open_dict[neighbor_pos] = neighbor_node
                heapq.heappush(open_list, neighbor_node)

    return None, nodes_expanded

def calculate_path_length_and_variance(path):
    """Calculate path length and variance"""
    if not path or len(path) < 2:
        return 0, 0
    distances = [
        math.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
        for i in range(len(path) - 1)
    ]
    return sum(distances), statistics.variance(distances) if len(distances) > 1 else 0

def visualize_path(img, path, start, end):
    """Visualize the path on the image"""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    for point in path:
        cv2.circle(img_color, point, 2, (0, 0, 255), -1)
    for i in range(len(path) - 1):
        cv2.line(img_color, path[i], path[i + 1], (0, 255, 0), 2)
    cv2.circle(img_color, start, 5, (255, 0, 0), -1)
    cv2.circle(img_color, end, 5, (0, 255, 0), -1)
    return img_color

def draw_grid(img, spacing=20):
    """Overlay a grid on the image for debugging"""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    for x in range(0, img.shape[1], spacing):
        cv2.line(img_color, (x, 0), (x, img.shape[0]), (200, 200, 200), 1)
    for y in range(0, img.shape[0], spacing):
        cv2.line(img_color, (0, y), (img.shape[1], y), (200, 200, 200), 1)
    return img_color

def execute_astar(obstacle_map, img_color, start, end):
    """Execute A* and display results on the same window."""
    start_time = time.time()
    path, nodes_expanded = astar(obstacle_map, start, end)
    end_time = time.time()

    if path:
        path_length, path_variance = calculate_path_length_and_variance(path)
        img_color_with_path = visualize_path(img_color, path, start, end)
        cv2.imshow('Select Start and Goal', img_color_with_path)  # Update the same window
        print(f"Path Length: {path_length:.2f}, Variance: {path_variance:.4f}, "
              f"Nodes Expanded: {nodes_expanded}, Time Taken: {end_time - start_time:.2f}s")
    else:
        print("No path found!")

def main():
    img_path = "Images/S1.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image. Check the path.")
        return

    # Create the obstacle map
    white_threshold = 200
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, obstacle_map = cv2.threshold(gray_img, white_threshold, 255, cv2.THRESH_BINARY_INV)

    img_color = draw_grid(img)

    choice = input("Enter '1' to input coordinates manually or '2' to select on the image: ").strip()
    if choice == '1':
        try:
            start_x = int(input("Enter start x-coordinate: "))
            start_y = int(input("Enter start y-coordinate: "))
            end_x = int(input("Enter end x-coordinate: "))
            end_y = int(input("Enter end y-coordinate: "))
            start = (start_x, start_y)
            end = (end_x, end_y)
            execute_astar(obstacle_map, img_color, start, end)
            cv2.waitKey(0)
        except ValueError:
            print("Invalid coordinates. Please enter valid integers.")
    elif choice == '2':
        points = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                color = (0, 255, 0) if len(points) == 1 else (0, 0, 255)
                cv2.circle(img_color, (x, y), 5, color, -1)
                cv2.imshow("Select Start and Goal", img_color)

                if len(points) == 2:
                    start, end = points
                    execute_astar(obstacle_map, img_color, start, end)

        cv2.imshow("Select Start and Goal", img_color)
        cv2.setMouseCallback("Select Start and Goal", mouse_callback)
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()