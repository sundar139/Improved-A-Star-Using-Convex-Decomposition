import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import triangulate, unary_union, polygonize
import triangle as tr
import math

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Error: Image not found!")
    return image

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        raise ValueError("No contours found!")
    return contours, hierarchy

def get_largest_contour(contours, hierarchy):
    max_area = 0
    map_boundary = None
    map_boundary_index = None
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                map_boundary = contour
                map_boundary_index = i
    if map_boundary is None:
        raise ValueError("Map boundary not found!")
    return map_boundary, map_boundary_index

def approximate_contour(contour, epsilon_factor=0.005):
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [tuple(pt[0]) for pt in approx]

def sort_vertices_clockwise(vertices):
    centroid = np.mean(vertices, axis=0)
    return sorted(
        vertices, key=lambda v: np.arctan2(v[1] - centroid[1], v[0] - centroid[0])
    )

def is_descendant(contour_index, ancestor_index, hierarchy):
    parent_index = hierarchy[0][contour_index][3]
    while parent_index != -1:
        if parent_index == ancestor_index:
            return True
        parent_index = hierarchy[0][parent_index][3]
    return False

def remove_duplicate_vertices(vertices, threshold=10):
    if not vertices:
        return []

    cleaned_vertices = [vertices[0]]
    for vertex in vertices[1:]:
        last = cleaned_vertices[-1]
        distance = math.hypot(vertex[0] - last[0], vertex[1] - last[1])
        if distance >= threshold:
            cleaned_vertices.append(vertex)
        else:
            cleaned_vertices[-1] = vertex

    if len(cleaned_vertices) > 1:
        first = cleaned_vertices[0]
        last = cleaned_vertices[-1]
        distance = math.hypot(first[0] - last[0], first[1] - last[1])
        if distance < threshold:
            cleaned_vertices.pop()

    return cleaned_vertices

def detect_obstacles(
    contours, hierarchy, map_boundary_index, map_boundary_area, vertices_dict
):
    obstacle_areas = []
    obstacle_count = 1
    for i, contour in enumerate(contours):
        if i == map_boundary_index:
            continue
        if is_descendant(i, map_boundary_index, hierarchy):
            area = cv2.contourArea(contour)
            if abs(area - map_boundary_area) < 20000:
                continue
            if any(abs(area - stored_area) < 1000 for stored_area in obstacle_areas):
                continue
            obstacle_areas.append(area)
            obstacle_vertices = approximate_contour(contour)
            obstacle_vertices = sort_vertices_clockwise(obstacle_vertices)
            obstacle_vertices = remove_duplicate_vertices(obstacle_vertices, threshold=10)
            if len(obstacle_vertices) < 3:
                print(f"Warning: Obstacle {obstacle_count} has less than 3 vertices after cleaning and will be skipped.")
                continue
            vertices_dict[f"Obstacle {obstacle_count}"] = obstacle_vertices
            obstacle_count += 1
    return vertices_dict

def classify_vertex_convexity(vertices):
    n = len(vertices)
    classifications = []
    for i in range(n):
        prev = np.array(vertices[(i - 1) % n])
        curr = np.array(vertices[i])
        next_v = np.array(vertices[(i + 1) % n])
        v1 = curr - prev
        v2 = next_v - curr
        cross_product = np.cross(v1, v2)
        classification = "Convex" if cross_product > 0 else "Concave"
        classifications.append((tuple(curr), classification))
    return classifications

def detect_polygon_orientation(vertices):
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return area >= 0

def perform_constrained_triangulation(map_boundary_vertices, vertices_dict):
    if not detect_polygon_orientation(map_boundary_vertices):
        map_boundary_vertices = map_boundary_vertices[::-1]
        print("Reversed map boundary vertex order for positive orientation.")

    holes = []
    for key in vertices_dict:
        if key.startswith("Obstacle"):
            obstacle_vertices = vertices_dict[key]
            if detect_polygon_orientation(obstacle_vertices):
                obstacle_vertices = obstacle_vertices[::-1]
                print(f"Reversed {key} vertex order for negative orientation.")
            holes.append(obstacle_vertices)

    segments = []
    points = []
    point_marker_dict = {}
    point_index = 0

    for i, vertex in enumerate(map_boundary_vertices):
        points.append(vertex)
        point_marker_dict[vertex] = point_index
        point_index += 1
    num_boundary_points = len(map_boundary_vertices)
    for i in range(num_boundary_points):
        idx1 = i
        idx2 = (i + 1) % num_boundary_points
        segments.append([idx1, idx2])

    for hole_vertices in holes:
        hole_point_indices = []
        for vertex in hole_vertices:
            if vertex not in point_marker_dict:
                points.append(vertex)
                point_marker_dict[vertex] = point_index
                point_index += 1
            hole_point_indices.append(point_marker_dict[vertex])
        num_hole_points = len(hole_vertices)
        for i in range(num_hole_points):
            idx1 = hole_point_indices[i]
            idx2 = hole_point_indices[(i + 1) % num_hole_points]
            segments.append([idx1, idx2])

    A = dict(vertices=np.array(points), segments=np.array(segments))

    hole_points = []
    for hole_vertices in holes:
        hole_polygon = Polygon(hole_vertices)
        x, y = hole_polygon.representative_point().coords[0]
        hole_points.append([x, y])

    A['holes'] = np.array(hole_points)

    B = tr.triangulate(A, 'p')

    sub_polygons = []
    for tri_indices in B.get('triangles', []):
        coords = [tuple(B['vertices'][idx]) for idx in tri_indices]
        sub_polygons.append(coords)
    return sub_polygons

def visualize_results(
    image, vertices_dict, convexity_results, sub_polygons, save_dir="Output", show_convexity=True
):
    import os

    os.makedirs(save_dir, exist_ok=True)

    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    for key, vertices in vertices_dict.items():
        color = (0, 255, 0) if key == "Map Boundary" else (255, 0, 0)
        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output_image, [pts], True, color, 2)

        if show_convexity:
            for vertex, classification in convexity_results[key]:
                marker_color = (0, 255, 0) if classification == "Convex" else (255, 0, 0)
                cv2.circle(output_image, vertex, 5, marker_color, -1)

    boundary_image_path = os.path.join(save_dir, "Boundary_and_Obstacles.png")
    boundary_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(boundary_image_path, boundary_bgr)
    print(f"Saved boundary and obstacles image to {boundary_image_path}")

    plt.figure(figsize=(12, 10))
    plt.imshow(output_image)
    #plt.title("Detected Map Boundary and Obstacles with Convex/Concave Vertices")
    plt.axis("off")
    plt.show()

    decomp_image = output_image.copy()
    for idx, poly in enumerate(sub_polygons):
        color = (255, 0, 0)
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(decomp_image, [pts], True, color, 1)

    decomposed_image_path = os.path.join(save_dir, "Convex_Decomposition.png")
    decomposed_bgr = cv2.cvtColor(decomp_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(decomposed_image_path, decomposed_bgr)
    #print(f"Saved convex decomposition image to {decomposed_image_path}")

    plt.figure(figsize=(12, 10))
    plt.imshow(decomp_image)
    #plt.title("Convex Sub-polygons after Decomposition (Edges Only)")
    plt.axis("off")
    plt.show()

    height, width, _ = output_image.shape
    sub_polygons_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    centroids = []
    all_edge_midpoints = []

    for poly in sub_polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(sub_polygons_image, [pts], True, (0, 0, 0), 1)

        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
            cv2.circle(sub_polygons_image, (cX, cY), 3, (0, 0, 255), -1)
        else:
            continue

        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            all_edge_midpoints.append((mid_x, mid_y))
            cv2.circle(sub_polygons_image, (mid_x, mid_y), 3, (0, 255, 0), -1)  # Green dot

    sub_polygons_image_path = os.path.join(save_dir, "Sub_Polygons_Edges_Centroids_Midpoints.png")
    sub_polygons_bgr = cv2.cvtColor(sub_polygons_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sub_polygons_image_path, sub_polygons_bgr)
    #print(f"Saved sub-polygons image with centroids and edge midpoints to {sub_polygons_image_path}")

    plt.figure(figsize=(12, 10))
    plt.imshow(sub_polygons_image)
    #plt.title("Convex Sub-polygons with Centroids (Red) and Edge Midpoints (Green)")
    plt.axis("off")
    plt.show()

    centroids_file_path = os.path.join(save_dir, "Centroids.txt")
    with open(centroids_file_path, 'w') as f:
        for idx, (cX, cY) in enumerate(centroids, start=1):
            f.write(f"Centroid {idx}: ({cX}, {cY})\n")
    #print(f"Saved centroids to {centroids_file_path}")

    edge_midpoints_file_path = os.path.join(save_dir, "Edge_Midpoints.txt")
    with open(edge_midpoints_file_path, 'w') as f:
        for idx, (mid_x, mid_y) in enumerate(all_edge_midpoints, start=1):
            f.write(f"Edge Midpoint {idx}: ({mid_x}, {mid_y})\n")
    #print(f"Saved edge midpoints to {edge_midpoints_file_path}")

def detect_and_decompose_map(image_path):
    try:
        image = read_image(image_path)
        contours, hierarchy = find_contours(image)

        map_boundary, map_boundary_index = get_largest_contour(contours, hierarchy)
        map_boundary_vertices = approximate_contour(map_boundary)
        map_boundary_vertices = sort_vertices_clockwise(map_boundary_vertices)

        vertices_dict = {"Map Boundary": map_boundary_vertices}
        map_boundary_area = cv2.contourArea(map_boundary)

        vertices_dict = detect_obstacles(
            contours, hierarchy, map_boundary_index, map_boundary_area, vertices_dict
        )

        convexity_results = {}
        for key, vertices in vertices_dict.items():
            convexity_results[key] = classify_vertex_convexity(vertices)

        print("\n=== Map Boundary and Obstacles Vertices ===")
        for key, vertices in vertices_dict.items():
            print(f"\n{key} Vertices ({len(vertices)} points):")
            for idx, vertex in enumerate(vertices, start=1):
                print(f"  Vertex {idx}: {vertex}")

        sub_polygons = perform_constrained_triangulation(map_boundary_vertices, vertices_dict)
        with open('Output/sub_polygons.txt', 'w') as f:
            print("\n=== Decomposed into the following convex sub-polygons (triangles) ===", file=f)
            for idx, poly in enumerate(sub_polygons, start=1):
                line = f"Sub-polygon {idx}: {poly}"
                print(line)
                f.write(line + '\n')

        visualize_results(
            image, vertices_dict, convexity_results, sub_polygons, save_dir="output", show_convexity=True
        )

        print("\n=== Decomposed into the following convex sub-polygons (triangles) ===")
        for idx, poly in enumerate(sub_polygons, start=1):
            print(f"Sub-polygon {idx}: {poly}")

    except Exception as e:
        print(str(e))

image_path = "Images/S1.png"
detect_and_decompose_map(image_path)
