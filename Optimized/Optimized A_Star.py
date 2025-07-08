import cv2
import numpy as np
import math
import heapq
import time
import os
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from scipy.interpolate import CubicSpline
from functools import lru_cache
import itertools

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
    obstacle_polygons = [Polygon(polygon) for polygon in sub_polygons]
    obstacle_union = unary_union(obstacle_polygons)
    obstacle_vertices = set()
    for polygon in sub_polygons:
        obstacle_vertices.update(polygon)
    points = list(obstacle_vertices)
    points.append(start_point)
    points.append(end_point)

    graph = {point: [] for point in points}

    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i >= j:
                continue

            line = LineString([point1, point2])

            # Check if the line intersects any obstacle
            if not obstacle_union.crosses(line):
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

    nodes_expanded = 0

    start_node = Node(start, g=0, h=heuristic(start, end))
    entry = (start_node.f, next(counter), start_node)
    heapq.heappush(open_heap, entry)
    open_entry_finder[start] = entry

    while open_heap:
        _, _, current_node = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current_node.position in closed_set:
            continue

        closed_set.add(current_node.position)

        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], nodes_expanded

        for neighbor_pos, cost in graph[current_node.position]:
            if neighbor_pos in closed_set:
                continue

            g_cost = current_node.g + cost
            h_cost = heuristic(neighbor_pos, end)
            f_cost = g_cost + h_cost

            if neighbor_pos in open_entry_finder:
                existing_entry = open_entry_finder[neighbor_pos]
                existing_node = existing_entry[2]
                if g_cost < existing_node.g:
                    neighbor_node = Node(neighbor_pos, g=g_cost, h=h_cost)
                    neighbor_node.parent = current_node
                    entry = (neighbor_node.f, next(counter), neighbor_node)
                    heapq.heappush(open_heap, entry)
                    open_entry_finder[neighbor_pos] = entry
            else:
                neighbor_node = Node(neighbor_pos, g=g_cost, h=h_cost)
                neighbor_node.parent = current_node
                entry = (neighbor_node.f, next(counter), neighbor_node)
                heapq.heappush(open_heap, entry)
                open_entry_finder[neighbor_pos] = entry
    return None, nodes_expanded

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
        cv2.line(img_color, (int(path[i][0]), int(path[i][1])), (int(path[i + 1][0]), int(path[i + 1][1])), (0, 0, 255), 2)  # Red line
    cv2.circle(img_color, (int(path[0][0]), int(path[0][1])), 5, (255, 0, 0), -1)  # Blue start
    cv2.circle(img_color, (int(path[-1][0]), int(path[-1][1])), 5, (0, 255, 0), -1)  # Green end
    cv2.imshow(title, img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_color

def smooth_path_bezier(path, obstacle_polygons):
    """Smooth the path using Bezier curves while avoiding obstacles."""
    path = np.array(path)
    num_points = max(100, len(path) * 10)

    t = np.linspace(0, 1, len(path))
    x = path[:, 0]
    y = path[:, 1]

    bezier_x = np.polyfit(t, x, deg=min(3, len(path)-1))
    bezier_y = np.polyfit(t, y, deg=min(3, len(path)-1))
    bezier_curve_x = np.poly1d(bezier_x)
    bezier_curve_y = np.poly1d(bezier_y)

    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = bezier_curve_x(t_smooth)
    y_smooth = bezier_curve_y(t_smooth)

    smoothed_path = []
    collision = False
    for i in range(len(x_smooth) - 1):
        point1 = (x_smooth[i], y_smooth[i])
        point2 = (x_smooth[i + 1], y_smooth[i + 1])
        line = LineString([point1, point2])

        intersects = False
        for obstacle in obstacle_polygons:
            if line.crosses(obstacle):
                intersects = True
                collision = True
                break
        if not intersects:
            smoothed_path.append(point1)
        else:
            break

    if collision:
        print("Collision detected during path smoothing with Bezier. Using original path.")
        return [tuple(p) for p in path]
    else:
        smoothed_path.append((x_smooth[-1], y_smooth[-1]))
        smoothed_path = [tuple(p) for p in smoothed_path]
        return smoothed_path

def main():
    img_path = "Images/S1.png"
    sub_polygons_file_path = "Output/sub_polygons.txt"

    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image. Check the path.")
        return

    sub_polygons = read_sub_polygons(sub_polygons_file_path)
    if not sub_polygons:
        print("Error: No sub-polygons found. Ensure sub_polygons.txt is properly formatted.")
        return

    obstacle_polygons = [Polygon(polygon) for polygon in sub_polygons]
    obstacle_union = unary_union(obstacle_polygons)

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
            path, nodes_expanded = astar_visibility_graph(graph, start_point, end_point)  # Adjusted to receive nodes_expanded
            end_time = time.time()

            if path:
                path_length, path_variance = calculate_path_length_and_variance(path)
                print(f"Path Length: {path_length:.2f}, Variance: {path_variance:.4f}, "
                      f"Nodes Expanded: {nodes_expanded}, Time Taken: {end_time - start_time:.2f}s")
                visualize_sparse_path(img_color, path, title="Initial Path")

                smoothed_path = smooth_path_bezier(path, obstacle_polygons)
                smoothed_length, smoothed_variance = calculate_path_length_and_variance(smoothed_path)
                print(f"Smoothed path! Length: {smoothed_length:.2f}, Variance: {smoothed_variance:.4f}")

                visualize_sparse_path(img_color, smoothed_path, title="Smoothed Path")
            else:
                print(f"No path found! Nodes Expanded: {nodes_expanded}")

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
                    path, nodes_expanded = astar_visibility_graph(graph, start_point, end_point)  # Adjusted to receive nodes_expanded
                    end_time = time.time()

                    if path:
                        path_length, path_variance = calculate_path_length_and_variance(path)
                        print(f"Initial path found! Length: {path_length:.2f}, Variance: {path_variance:.4f}, "
                              f"Nodes Expanded: {nodes_expanded}, Time Taken: {end_time - start_time:.2f}s")
                        visualize_sparse_path(img_color, path, title="Initial Path")

                        smoothed_path = smooth_path_bezier(path, obstacle_polygons)
                        smoothed_length, smoothed_variance = calculate_path_length_and_variance(smoothed_path)
                        print(f"Smoothed path! Length: {smoothed_length:.2f}, Variance: {smoothed_variance:.4f}")

                        visualize_sparse_path(img_color, smoothed_path, title="Smoothed Path")
                    else:
                        print(f"No path found! Nodes Expanded: {nodes_expanded}")

        cv2.imshow("Select Start and Goal", img_color)
        cv2.setMouseCallback("Select Start and Goal", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()
