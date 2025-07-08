import cv2
import numpy as np
import math
import heapq
import time
import os
from shapely.geometry import Polygon, Point

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

def heuristic(a, b):
    """Euclidean distance heuristic"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def read_centroids(centroids_file_path):
    """Read centroids from the file"""
    centroids = []
    with open(centroids_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                coord_str = parts[1].strip().strip('()')
                x_str, y_str = coord_str.split(',')
                x, y = int(float(x_str.strip())), int(float(y_str.strip()))
                centroids.append((x, y))
    return centroids

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

def build_sparse_graph(centroids, sub_polygons, start_point, end_point):
    """Build a graph where nodes are centroids, start, and goal; edges are based on adjacency."""
    graph = {centroid: [] for centroid in centroids}
    graph[start_point] = []
    graph[end_point] = []

    # Create a mapping from centroid to its sub-polygon
    centroid_to_polygon = {}
    for centroid, polygon in zip(centroids, sub_polygons):
        centroid_to_polygon[centroid] = polygon

    # For each sub-polygon, find adjacent sub-polygons and connect their centroids
    for i, poly1 in enumerate(sub_polygons):
        centroid1 = centroids[i]
        edges1 = set(tuple(sorted(edge)) for edge in zip(poly1, poly1[1:] + poly1[:1]))
        for j in range(i + 1, len(sub_polygons)):
            poly2 = sub_polygons[j]
            centroid2 = centroids[j]
            edges2 = set(tuple(sorted(edge)) for edge in zip(poly2, poly2[1:] + poly2[:1]))
            # If they share an edge, connect their centroids
            if edges1 & edges2:
                distance = heuristic(centroid1, centroid2)
                graph[centroid1].append((centroid2, distance))
                graph[centroid2].append((centroid1, distance))

    # Connect start_point to centroids (unidirectional: start_point -> centroids)
    start_connected = False
    for centroid, polygon in centroid_to_polygon.items():
        poly_shape = Polygon(polygon)
        if poly_shape.contains(Point(start_point)):
            distance = heuristic(start_point, centroid)
            graph[start_point].append((centroid, distance))
            start_connected = True

    # If start_point is not inside any polygon, connect to the nearest centroid
    if not start_connected:
        nearest_centroid = min(centroids, key=lambda c: heuristic(c, start_point))
        distance = heuristic(start_point, nearest_centroid)
        graph[start_point].append((nearest_centroid, distance))

    # Connect centroids to end_point (unidirectional: centroids -> end_point)
    end_connected = False
    for centroid, polygon in centroid_to_polygon.items():
        poly_shape = Polygon(polygon)
        if poly_shape.contains(Point(end_point)):
            distance = heuristic(centroid, end_point)
            graph[centroid].append((end_point, distance))
            end_connected = True

    # If end_point is not inside any polygon, connect nearest centroid to end_point
    if not end_connected:
        nearest_centroid = min(centroids, key=lambda c: heuristic(c, end_point))
        distance = heuristic(nearest_centroid, end_point)
        graph[nearest_centroid].append((end_point, distance))

    return graph

def sparse_astar(graph, start, end):
    """A* pathfinding algorithm on a sparse graph."""
    open_list = []
    open_dict = {}
    closed_set = set()

    start_node = Node(start, g=0, h=heuristic(start, end))
    heapq.heappush(open_list, start_node)
    open_dict[start] = start_node

    while open_list:
        current_node = heapq.heappop(open_list)
        del open_dict[current_node.position]
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
                continue

            g_cost = current_node.g + cost
            h_cost = heuristic(neighbor_pos, end)
            f_cost = g_cost + h_cost

            if neighbor_pos in open_dict:
                existing_node = open_dict[neighbor_pos]
                if g_cost < existing_node.g:
                    existing_node.g = g_cost
                    existing_node.h = h_cost
                    existing_node.f = f_cost
                    existing_node.parent = current_node
            else:
                neighbor_node = Node(neighbor_pos, g=g_cost, h=h_cost)
                neighbor_node.parent = current_node
                open_dict[neighbor_pos] = neighbor_node
                heapq.heappush(open_list, neighbor_node)
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

def visualize_sparse_path(img, path):
    """Visualize the path on the image"""
    img_color = img.copy()
    for i in range(len(path) - 1):
        cv2.line(img_color, path[i], path[i + 1], (0, 0, 255), 2)  # Red line
    cv2.circle(img_color, path[0], 5, (255, 0, 0), -1)  # Blue start
    cv2.circle(img_color, path[-1], 5, (0, 255, 0), -1)  # Green end
    return img_color

def execute_sparse_astar(graph, start_point, end_point, img_color):
    """Execute sparse A* and display results."""
    start_time = time.time()
    path = sparse_astar(graph, start_point, end_point)
    end_time = time.time()

    if path:
        path_length, path_variance = calculate_path_length_and_variance(path)
        img_color_with_path = visualize_sparse_path(img_color, path)
        cv2.imshow('Select Start and Goal', img_color_with_path)  # Update the same window
        print(f"Path found! Length: {path_length:.2f}, Variance: {path_variance:.4f}, "
              f"Time Taken: {end_time - start_time:.2f}s")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No path found!")

def main():
    img_path = "Images/S1.png"  # Original image path
    centroids_file_path = "output/centroids.txt"
    sub_polygons_file_path = "output/sub_polygons.txt"

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image. Check the path.")
        return

    # Read centroids and sub-polygons
    centroids = read_centroids(centroids_file_path)
    if not centroids:
        print("Error: No centroids found. Ensure centroids.txt is properly formatted.")
        return

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

            graph = build_sparse_graph(centroids, sub_polygons, start_point, end_point)
            execute_sparse_astar(graph, start_point, end_point, img_color)
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

                    graph = build_sparse_graph(centroids, sub_polygons, start_point, end_point)
                    execute_sparse_astar(graph, start_point, end_point, img_color)

        cv2.imshow("Select Start and Goal", img_color)
        cv2.setMouseCallback("Select Start and Goal", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()
