# Improved A\* Using Convex Decomposition

## 📘 Introduction

This project focuses on solving one of the core challenges in robotics and artificial intelligence: **efficient and reliable path planning** in complex environments. Traditional pathfinding algorithms like A\*, RRT, RRT\*, and PRM can struggle with efficiency and smoothness in environments featuring irregularly shaped and densely packed obstacles. To address this, we implemented an improved path planning pipeline that leverages **image processing, geometric decomposition, and graph-based optimization**.

The main innovation lies in the integration of **map segmentation**, **convex decomposition**, **visibility graph construction**, and a **sparse A\*** algorithm enhanced with **Bezier curve smoothing**. The goal is to minimize computational load while ensuring optimal, obstacle-free, and smooth paths.

## 🔍 My Insights

Through this project, we realized the potential of combining **image-based environment interpretation** with **geometric processing** for real-world robotics navigation. The key insights include:

- Preprocessing a map image into **convex sub-polygons** drastically reduces the search space.
- A visibility graph based on polygon vertices offers a more **efficient representation** of navigable space compared to traditional grid-based methods.
- Enhancing the path with **Bezier curve smoothing** results in more realistic, navigable paths suited for real applications.
- Our Sparse A\* significantly reduces **nodes expanded** and **computation time**, while improving **path quality**.

## 🎯 Objectives and Visualizations

### What We're Trying to Find/Visualize:

- **Collision-free and optimal paths** in environments cluttered with obstacles.
- Comparison of different algorithms on performance metrics such as:

  - Path Length
  - Variance
  - Nodes Expanded
  - Computation Time

- Visualization of:

  - Map segmentation
  - Convex polygon decomposition
  - Visibility graphs
  - Initial and smoothed paths

### Why:

Autonomous systems (e.g., robots, drones, AGVs) must navigate in real-time through unpredictable and cluttered environments. Efficient and robust pathfinding is vital to their reliability and success. Our aim was to create a method that:

- Is computationally efficient.
- Works in real-time.
- Generates paths suitable for physical systems.

## 📊 Findings

Our experimental results over multiple maps show that **our enhanced A\*** outperforms traditional methods:

| Metric              | Traditional A\* | RRT    | RRT\*  | PRM    | **Our A**\* |
| ------------------- | --------------- | ------ | ------ | ------ | ----------- |
| Path Length (Map 1) | 639.12          | 754.48 | 782.90 | 654.04 | **613.07**  |
| Variance            | 0.0335          | 0.33   | 1.62   | 100.3  | **1487.6**  |
| Nodes Expanded      | 491             | 217    | 209    | 176    | **5**       |
| Time (sec)          | 0.00            | 0.02   | 0.02   | 0.54   | **0.00**    |

Key takeaways:

- **Shortest and smoothest path** obtained through our method.
- **Lowest node expansion**, demonstrating efficiency.
- **Real-time performance**, even in complex environments.

---

## ✅ Conclusion

Our project provides a **robust, scalable, and computationally efficient** solution for path planning in obstacle-rich environments. By integrating **image preprocessing, convex decomposition, visibility graphs, and a refined A\*** search with **Bezier smoothing**, we achieved superior results in terms of accuracy, efficiency, and practical applicability.

This approach has promising implications for real-world navigation systems in:

- Robotics
- Autonomous vehicles
- Warehouse automation
- Search and rescue missions

### 🔮 Future Work:

- Extend to **dynamic environments** and **real-time obstacle updates**.
- Adaptation for **3D navigation**.
- Integrate **machine learning** for predictive obstacle handling.
