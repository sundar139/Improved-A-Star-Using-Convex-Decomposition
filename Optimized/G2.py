import cv2
import numpy as np
import math
import heapq
import time
import os
from shapely.geometry import Polygon, Point, LineString
from functools import lru_cache
import itertools

class Node:
    """A node class for A* Pathfinding"""
    def __init__(self, position, g=0, h=0):
        self.position = position  # (x, y)
        self.g = g  # Cost from start node
        self.h = h  # Heuristic (estimated cost to end)
        self.f = g + h  # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

@lru_cache(maxsize=None)
def heuristic(a, b):
    """Euclidean distance heuristic"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def read_sub_polygons(sub_polygons_file_path):
    """Read sub-polygons from the file"""
    sub_polygons = []
    with open(sub_polygons_file_path, 'r') as f:
        for line in f:
            if line.startswith('Sub-polygon'):
                parts = line.strip().split(':', 1)
                coords_str = parts[1].strip().strip('[]')
                coords = []
                for coord_pair in coords_str.split('), ('):
                    coord_pair = coord_pair.strip('() ')
                    if coord_pair:
                        x_str, y_str = coord_pair.split(',')
                        x, y = float(x_str.strip()), float(y_str.strip())
                        coords.append((int(x), int(y)))
                sub_polygons.append(coords)
    return sub_polygons

def build_visibility_graph(sub_polygons, start_point, end_point):
    """
    Build a visibility graph using obstacle vertices, start, and end points.
    """
    # Collect all vertices from obstacles
    obstacle_polygons = [Polygon(polygon) for polygon in sub_polygons]
    obstacle_vertices = set()
    for polygon in sub_polygons:
        obstacle_vertices.update(polygon)
    # Add start and end points
    points = list(obstacle_vertices)
    points.append(start_point)
    points.append(end_point)

    # Initialize the graph
    graph = {point: [] for point in points}

    # Build the visibility graph
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i >= j:
                continue  # Avoid duplicate edges and self-loops

            line = LineString([point1, point2])

            # Check if the line intersects any obstacle
            is_visible = True
            for obstacle in obstacle_polygons:
                if line.crosses(obstacle) and not line.touches(obstacle):
                    is_visible = False
                    break

            if is_visible:
                distance = heuristic(point1, point2)
                graph[point1].append((point2, distance))
                graph[point2].append((point1, distance))

    return graph

def astar_visibility_graph(graph, start, end):
    """A* pathfinding algorithm on the visibility graph."""
    counter = itertools.count()
    open_heap = []
    open_entry_finder = {}
    closed_set = set()

    start_node = Node(start, g=0, h=heuristic(start, end))
    entry = (start_node.f, next(counter), start_node)
    heapq.heappush(open_heap, entry)
    open_entry_finder[start] = entry

    while open_heap:
        _, _, current_node = heapq.heappop(open_heap)

        if current_node.position in closed_set:
            continue  # Skip nodes that have already been processed

        closed_set.add(current_node.position)

        if current_node.position == end:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Reverse path

        for neighbor_pos, cost in graph[current_node.position]:
            if neighbor_pos in closed_set:
                continue  # Skip neighbors that are already processed

            g_cost = current_node.g + cost
            h_cost = heuristic(neighbor_pos, end)
            f_cost = g_cost + h_cost

            if neighbor_pos in open_entry_finder:
                existing_entry = open_entry_finder[neighbor_pos]
                existing_node = existing_entry[2]
                if g_cost < existing_node.g:
                    # Found a better path to this neighbor
                    neighbor_node = Node(neighbor_pos, g=g_cost, h=h_cost)
                    neighbor_node.parent = current_node
                    entry = (neighbor_node.f, next(counter), neighbor_node)
                    heapq.heappush(open_heap, entry)
                    open_entry_finder[neighbor_pos] = entry
            else:
                # New neighbor, add to open list
                neighbor_node = Node(neighbor_pos, g=g_cost, h=h_cost)
                neighbor_node.parent = current_node
                entry = (neighbor_node.f, next(counter), neighbor_node)
                heapq.heappush(open_heap, entry)
                open_entry_finder[neighbor_pos] = entry
    return None  # No path found

def calculate_path_length_and_variance(path):
    """Calculate path length and variance"""
    if not path or len(path) < 2:
        return 0, 0
    distances = [
        heuristic(path[i], path[i + 1])
        for i in range(len(path) - 1)
    ]
    return sum(distances), np.var(distances) if len(distances) > 1 else 0

def visualize_sparse_path(img, path, title="Path Found"):
    """Visualize the path on the image"""
    img_color = img.copy()
    for i in range(len(path) - 1):
        cv2.line(img_color, path[i], path[i + 1], (0, 0, 255), 2)  # Red line
    cv2.circle(img_color, path[0], 5, (255, 0, 0), -1)  # Blue start
    cv2.circle(img_color, path[-1], 5, (0, 255, 0), -1)  # Green end
    cv2.imshow(title, img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_color

def main():
    img_path = "Images/S1.png"
    sub_polygons_file_path = "output/sub_polygons.txt"

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image. Check the path.")
        return

    # Read sub-polygons
    sub_polygons = read_sub_polygons(sub_polygons_file_path)
    if not sub_polygons:
        print("Error: No sub-polygons found. Ensure sub_polygons.txt is properly formatted.")
        return

    img_color = img.copy()

    choice = input("Enter '1' to input coordinates manually or '2' to select on the image: ").strip()
    if choice == '1':
        try:
            start_x = int(input("Enter start x-coordinate: "))
            start_y = int(input("Enter start y-coordinate: "))
            end_x = int(input("Enter end x-coordinate: "))
            end_y = int(input("Enter end y-coordinate: "))
            start_point = (start_x, start_y)
            end_point = (end_x, end_y)

            graph = build_visibility_graph(sub_polygons, start_point, end_point)
            start_time = time.time()
            path = astar_visibility_graph(graph, start_point, end_point)
            end_time = time.time()

            if path:
                path_length, path_variance = calculate_path_length_and_variance(path)
                print(f"Path found! Length: {path_length:.2f}, Variance: {path_variance:.4f}, "
                      f"Time Taken: {end_time - start_time:.2f}s")
                visualize_sparse_path(img_color, path, title="Optimized Path")
            else:
                print("No path found!")

        except ValueError:
            print("Invalid coordinates. Please enter valid integers.")
    elif choice == '2':
        points = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                color = (255, 0, 0) if len(points) == 1 else (0, 255, 0)
                cv2.circle(img_color, (x, y), 5, color, -1)
                cv2.imshow("Select Start and Goal", img_color)

                if len(points) == 2:
                    start_point, end_point = points

                    graph = build_visibility_graph(sub_polygons, start_point, end_point)
                    start_time = time.time()
                    path = astar_visibility_graph(graph, start_point, end_point)
                    end_time = time.time()

                    if path:
                        path_length, path_variance = calculate_path_length_and_variance(path)
                        print(f"Path found! Length: {path_length:.2f}, Variance: {path_variance:.4f}, "
                              f"Time Taken: {end_time - start_time:.2f}s")
                        visualize_sparse_path(img_color, path, title="Optimized Path")
                    else:
                        print("No path found!")

        cv2.imshow("Select Start and Goal", img_color)
        cv2.setMouseCallback("Select Start and Goal", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()
