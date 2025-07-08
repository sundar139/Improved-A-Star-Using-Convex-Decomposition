import cv2
import numpy as np
import math
import heapq
import time
import os
from shapely.geometry import Polygon, Point
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
    """
    Build a graph where nodes are centroids and edge midpoints, start, and goal;
    edges are based on adjacency.
    """
    # Initialize graph nodes
    nodes = set()
    nodes.update(centroids)
    nodes.add(start_point)
    nodes.add(end_point)

    # Create mappings
    centroid_to_polygon = {}
    centroid_to_edge_midpoints = {}
    edge_midpoint_to_centroids = {}

    # Process each sub-polygon
    for centroid, polygon in zip(centroids, sub_polygons):
        centroid_to_polygon[centroid] = polygon
        # Compute edge midpoints
        edge_midpoints = []
        num_vertices = len(polygon)
        for i in range(num_vertices):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % num_vertices]
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            edge_midpoint = (mid_x, mid_y)
            edge_midpoints.append(edge_midpoint)
            # Map edge midpoint to centroids
            if edge_midpoint in edge_midpoint_to_centroids:
                edge_midpoint_to_centroids[edge_midpoint].add(centroid)
            else:
                edge_midpoint_to_centroids[edge_midpoint] = {centroid}
        centroid_to_edge_midpoints[centroid] = edge_midpoints
        # Add edge midpoints to nodes
        nodes.update(edge_midpoints)

    # Initialize the graph with empty adjacency lists
    graph = {node: [] for node in nodes}

    # Connect centroids to their edge midpoints
    for centroid, edge_midpoints in centroid_to_edge_midpoints.items():
        for edge_midpoint in edge_midpoints:
            distance = heuristic(centroid, edge_midpoint)
            graph[centroid].append((edge_midpoint, distance))
            graph[edge_midpoint].append((centroid, distance))

    # Connect adjacent edge midpoints within the same sub-polygon
    for edge_midpoints in centroid_to_edge_midpoints.values():
        num_midpoints = len(edge_midpoints)
        for i in range(num_midpoints):
            midpoint1 = edge_midpoints[i]
            midpoint2 = edge_midpoints[(i + 1) % num_midpoints]
            distance = heuristic(midpoint1, midpoint2)
            graph[midpoint1].append((midpoint2, distance))
            graph[midpoint2].append((midpoint1, distance))

    # Connect shared edge midpoints between adjacent sub-polygons
    for edge_midpoint, centroid_set in edge_midpoint_to_centroids.items():
        centroids_list = list(centroid_set)
        for i in range(len(centroids_list)):
            for j in range(i + 1, len(centroids_list)):
                centroid1 = centroids_list[i]
                centroid2 = centroids_list[j]
                distance = heuristic(centroid1, centroid2)
                # Connect centroids via the shared edge midpoint
                graph[centroid1].append((edge_midpoint, heuristic(centroid1, edge_midpoint)))
                graph[edge_midpoint].append((centroid1, heuristic(edge_midpoint, centroid1)))
                graph[centroid2].append((edge_midpoint, heuristic(centroid2, edge_midpoint)))
                graph[edge_midpoint].append((centroid2, heuristic(edge_midpoint, centroid2)))

    # Connect start_point to the graph
    start_connected = False
    for centroid, polygon in centroid_to_polygon.items():
        poly_shape = Polygon(polygon)
        if poly_shape.contains(Point(start_point)):
            # Connect start_point to centroid
            distance = heuristic(start_point, centroid)
            graph[start_point].append((centroid, distance))
            graph[centroid].append((start_point, distance))
            # Connect start_point to edge midpoints
            for edge_midpoint in centroid_to_edge_midpoints[centroid]:
                distance = heuristic(start_point, edge_midpoint)
                graph[start_point].append((edge_midpoint, distance))
                graph[edge_midpoint].append((start_point, distance))
            start_connected = True
            break  # Assuming start_point can be in only one sub-polygon
    if not start_connected:
        # Connect to the nearest node (centroid or edge midpoint)
        nearest_node = min(nodes, key=lambda n: heuristic(n, start_point))
        distance = heuristic(start_point, nearest_node)
        graph[start_point].append((nearest_node, distance))
        graph[nearest_node].append((start_point, distance))

    # Connect end_point to the graph
    end_connected = False
    for centroid, polygon in centroid_to_polygon.items():
        poly_shape = Polygon(polygon)
        if poly_shape.contains(Point(end_point)):
            # Connect end_point to centroid
            distance = heuristic(end_point, centroid)
            graph[end_point].append((centroid, distance))
            graph[centroid].append((end_point, distance))
            # Connect end_point to edge midpoints
            for edge_midpoint in centroid_to_edge_midpoints[centroid]:
                distance = heuristic(end_point, edge_midpoint)
                graph[end_point].append((edge_midpoint, distance))
                graph[edge_midpoint].append((end_point, distance))
            end_connected = True
            break
    if not end_connected:
        # Connect to the nearest node (centroid or edge midpoint)
        nearest_node = min(nodes, key=lambda n: heuristic(n, end_point))
        distance = heuristic(end_point, nearest_node)
        graph[end_point].append((nearest_node, distance))
        graph[nearest_node].append((end_point, distance))

    return graph

def sparse_astar(graph, start, end):
    """A* pathfinding algorithm on a sparse graph."""
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
