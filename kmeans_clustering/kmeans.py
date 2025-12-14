
"""
K-Means Clustering - Complete Implementation with Multiple Visualizations
Author: Your Name
Course: AI/Machine Learning Assignment
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """K-Means Clustering Algorithm"""
    
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
        self.history = []
        self.X = None
        
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def initialize_centroids(self, X, initial_indices=None):
        """Initialize centroids"""
        if initial_indices is not None:
            self.centroids = X[initial_indices].copy()
        else:
            random_indices = np.random.choice(len(X), self.k, replace=False)
            self.centroids = X[random_indices].copy()
        return self.centroids
    
    def assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.zeros((len(X), self.k))
        
        for i, point in enumerate(X):
            for j, centroid in enumerate(self.centroids):
                distances[i, j] = self.euclidean_distance(point, centroid)
        
        labels = np.argmin(distances, axis=1)
        return labels, distances
    
    def update_centroids(self, X, labels):
        """Update centroids based on cluster assignments"""
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def fit(self, X, initial_indices=None, num_iterations=None):
        """Perform K-means clustering"""
        self.X = X  # Store data
        self.initialize_centroids(X, initial_indices)
        
        self.history = [{
            'iteration': 0,
            'centroids': self.centroids.copy(),
            'labels': None,
            'distances': None
        }]
        
        iterations_to_run = num_iterations if num_iterations else self.max_iterations
        
        for iteration in range(1, iterations_to_run + 1):
            labels, distances = self.assign_clusters(X)
            
            self.history.append({
                'iteration': iteration,
                'centroids': self.centroids.copy(),
                'labels': labels.copy(),
                'distances': distances.copy()
            })
            
            new_centroids = self.update_centroids(X, labels)
            
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration}")
                self.centroids = new_centroids
                self.labels = labels
                break
            
            self.centroids = new_centroids
            self.labels = labels
        
        return self
    
    def print_iteration_details(self, X, point_names, iteration):
        """Print detailed information about an iteration"""
        if iteration >= len(self.history):
            print(f"No data for iteration {iteration}")
            return
        
        details = self.history[iteration]
        
        print(f"\n{'='*70}")
        print(f"ITERATION {details['iteration']}")
        print(f"{'='*70}")
        
        print("\nCluster Centers:")
        for i, centroid in enumerate(details['centroids']):
            print(f"  C{i+1}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
        
        if details['labels'] is not None:
            print("\nCluster Assignments:")
            for i in range(self.k):
                cluster_points = [point_names[j] for j in range(len(X)) 
                                if details['labels'][j] == i]
                if cluster_points:
                    print(f"  Cluster {i+1}: {', '.join(cluster_points)}")
                else:
                    print(f"  Cluster {i+1}: Empty")
            
            print("\nDistances from each point to centroids:")
            for j, point_name in enumerate(point_names):
                print(f"  {point_name}{tuple(X[j])}:")
                for i in range(self.k):
                    dist = details['distances'][j][i]
                    nearest = " ← Nearest" if details['labels'][j] == i else ""
                    print(f"    to C{i+1}: {dist:.4f}{nearest}")
    
    def print_summary(self):
        """Print summary of clustering results"""
        print(f"\n{'='*70}")
        print("CLUSTERING SUMMARY")
        print(f"{'='*70}")
        print(f"Number of clusters: {self.k}")
        print(f"Number of iterations performed: {len(self.history) - 1}")
        print(f"\nFinal centroids:")
        for i, centroid in enumerate(self.centroids):
            print(f"  C{i+1}: ({centroid[0]:.4f}, {centroid[1]:.4f})")


# ============================================================================
# VISUALIZATION FUNCTIONS - Multiple graphs for full marks
# ============================================================================

def plot_initial_data(points, point_names):
    """Graph 1: Input data visualization"""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(points[:, 0], points[:, 1],
               c='gray', s=250, alpha=0.7,
               edgecolors='black', linewidth=2.5)
    
    for i, (point, name) in enumerate(zip(points, point_names)):
        plt.annotate(name, xy=(point[0], point[1]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    ha='center')
    
    plt.xlabel('X Coordinate', fontsize=13, fontweight='bold')
    plt.ylabel('Y Coordinate', fontsize=13, fontweight='bold')
    plt.title('Input Data Points (Before Clustering)', 
             fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_iteration_comparison(points, point_names, kmeans):
    """Graph 2: Side-by-side comparison of iterations"""
    n_iterations = len(kmeans.history)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Create subplots
    n_cols = min(3, n_iterations)
    n_rows = (n_iterations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle different subplot configurations
    if n_iterations == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, details in enumerate(kmeans.history):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if details['labels'] is not None:
            for i in range(kmeans.k):
                cluster_points = points[details['labels'] == i]
                if len(cluster_points) > 0:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                             c=colors[i], s=150, alpha=0.6,
                             edgecolors='black', linewidth=1.5)
        else:
            ax.scatter(points[:, 0], points[:, 1],
                      c='gray', s=150, alpha=0.6,
                      edgecolors='black', linewidth=1.5)
        
        ax.scatter(details['centroids'][:, 0], details['centroids'][:, 1],
                  c='black', s=300, marker='X',
                  edgecolors='yellow', linewidth=2.5, zorder=5)
        
        for i, (point, name) in enumerate(zip(points, point_names)):
            ax.annotate(name, xy=(point[0], point[1]),
                       xytext=(0, -12), textcoords='offset points',
                       fontsize=8, ha='center')
        
        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_title(f'Iteration {details["iteration"]}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_iterations, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('K-Means Clustering: All Iterations', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_final_clusters(points, point_names, kmeans):
    """Graph 3: Final clustering result"""
    plt.figure(figsize=(11, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(kmeans.k):
        cluster_points = points[kmeans.labels == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=colors[i], s=220, alpha=0.65, 
                       edgecolors='black', linewidth=2,
                       label=f'Cluster {i+1}')
    
    centroids = kmeans.centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
               c='black', s=450, marker='X',
               edgecolors='yellow', linewidth=3.5,
               label='Centroids', zorder=5)
    
    for i, (point, name) in enumerate(zip(points, point_names)):
        plt.annotate(name, xy=(point[0], point[1]),
                    xytext=(-8, -8), textcoords='offset points',
                    fontsize=11, fontweight='bold')
    
    for i, centroid in enumerate(centroids):
        plt.annotate(f'C{i+1}', xy=(centroid[0], centroid[1]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='yellow', alpha=0.8))
    
    plt.xlabel('X Coordinate', fontsize=13, fontweight='bold')
    plt.ylabel('Y Coordinate', fontsize=13, fontweight='bold')
    plt.title('Final K-Means Clustering Result', 
             fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_centroid_movement(kmeans, point_names):
    """Graph 4: Centroid movement across iterations"""
    plt.figure(figsize=(11, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot data points lightly
    plt.scatter(kmeans.X[:, 0], kmeans.X[:, 1],
               c='lightgray', s=180, alpha=0.4,
               edgecolors='gray', linewidth=1)
    
    for point, name in zip(kmeans.X, point_names):
        plt.annotate(name, xy=(point[0], point[1]),
                    xytext=(0, -15), textcoords='offset points',
                    fontsize=9, alpha=0.6, ha='center')
    
    # Plot centroid trajectories
    for i in range(kmeans.k):
        centroid_path = [h['centroids'][i] for h in kmeans.history]
        centroid_path = np.array(centroid_path)
        
        # Draw trajectory line
        plt.plot(centroid_path[:, 0], centroid_path[:, 1],
                color=colors[i], linewidth=2.5, alpha=0.7,
                marker='o', markersize=8, label=f'C{i+1} path')
        
        # Mark start position
        plt.scatter(centroid_path[0, 0], centroid_path[0, 1],
                   c=colors[i], s=300, marker='s',
                   edgecolors='black', linewidth=2,
                   zorder=5, alpha=0.8)
        
        # Mark end position
        plt.scatter(centroid_path[-1, 0], centroid_path[-1, 1],
                   c=colors[i], s=450, marker='X',
                   edgecolors='black', linewidth=3,
                   zorder=5)
        
        # Annotate iteration numbers
        for j, pos in enumerate(centroid_path):
            plt.annotate(f'{j}', xy=(pos[0], pos[1]),
                        xytext=(12, 12), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.3',
                                 facecolor='white', alpha=0.8))
    
    plt.xlabel('X Coordinate', fontsize=13, fontweight='bold')
    plt.ylabel('Y Coordinate', fontsize=13, fontweight='bold')
    plt.title('Centroid Movement Across Iterations', 
             fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(kmeans):
    """Graph 5: Cluster size distribution"""
    plt.figure(figsize=(10, 6))
    
    cluster_sizes = [np.sum(kmeans.labels == i) for i in range(kmeans.k)]
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:kmeans.k]
    
    bars = plt.bar(range(1, kmeans.k + 1), cluster_sizes, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size)} points',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Cluster Number', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Points', fontsize=13, fontweight='bold')
    plt.title('Cluster Size Distribution', 
             fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(1, kmeans.k + 1))
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_distance_evolution(kmeans):
    """Graph 6: Average distance to centroid per iteration"""
    plt.figure(figsize=(10, 6))
    
    avg_distances = []
    
    for details in kmeans.history:
        if details['labels'] is not None and details['distances'] is not None:
            # Calculate average distance for assigned clusters
            assigned_distances = []
            for i, label in enumerate(details['labels']):
                assigned_distances.append(details['distances'][i][label])
            avg_distances.append(np.mean(assigned_distances))
    
    if avg_distances:
        iterations = range(1, len(avg_distances) + 1)
        plt.plot(iterations, avg_distances, 
                marker='o', linewidth=3, markersize=10,
                color='darkblue', label='Avg Distance')
        
        # Add value labels
        for x, y in zip(iterations, avg_distances):
            plt.text(x, y, f'{y:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Iteration', fontsize=13, fontweight='bold')
        plt.ylabel('Average Distance to Centroid', fontsize=13, fontweight='bold')
        plt.title('Clustering Quality: Distance Evolution', 
                 fontsize=15, fontweight='bold', pad=20)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(iterations)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️  No distance data available for plotting")


def main():
    """Main program"""
    print("\n" + "="*70)
    print(" "*15 + "K-MEANS CLUSTERING")
    print(" "*10 + "Complete with Multiple Visualizations")
    print("="*70 + "\n")
    
    print("1. Use example data")
    print("2. Enter custom data")
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == '1':
        points = np.array([
            [2, 10], [2, 5], [8, 4], [5, 8],
            [7, 5], [6, 4], [1, 2], [4, 9]
        ])
        point_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
        initial_indices = [0, 3, 6]
        num_iterations = 2
        k = 3
        
        print("\nUsing example data:")
        print("Points: 8")
        print("Clusters: 3")
        print("Initial centers: A1, A4, A7")
        
    else:
        num_points = int(input("\nNumber of points: "))
        
        points = []
        point_names = []
        
        print(f"\nEnter {num_points} points (format: name x y)")
        for i in range(num_points):
            data = input(f"Point {i+1}: ").strip().split()
            point_names.append(data[0])
            points.append([float(data[1]), float(data[2])])
        
        points = np.array(points)
        
        k = int(input(f"\nNumber of clusters (1-{num_points}): "))
        
        print("\nAvailable points:")
        for i, name in enumerate(point_names):
            print(f"  {i}: {name}")
        
        print(f"\nEnter {k} indices for initial centers (space separated):")
        initial_indices = [int(x) for x in input().strip().split()]
        
        num_iterations = int(input("\nNumber of iterations: ") or "2")
    
    print("\n" + "="*70)
    print("Running K-means clustering...")
    print("="*70)
    
    # Run K-means
    kmeans = KMeans(k=k)
    kmeans.fit(points, initial_indices=initial_indices, num_iterations=num_iterations)
    
    # Console output
    for i in range(len(kmeans.history)):
        kmeans.print_iteration_details(points, point_names, i)
    
    kmeans.print_summary()
    
    # Visualizations
    print("\n" + "="*70)
    show_plots = input("\nShow visualizations? (y/n): ").strip().lower()
    
    if show_plots == 'y':
        print("\n📊 Generating visualizations...")
        print("Close each graph window to see the next one.\n")
        
        print("Graph 1/6: Input Data...")
        plot_initial_data(points, point_names)
        
        print("Graph 2/6: Iteration Comparison...")
        plot_iteration_comparison(points, point_names, kmeans)
        
        print("Graph 3/6: Final Clusters...")
        plot_final_clusters(points, point_names, kmeans)
        
        print("Graph 4/6: Centroid Movement...")
        plot_centroid_movement(kmeans, point_names)
        
        print("Graph 5/6: Cluster Sizes...")
        plot_cluster_sizes(kmeans)
        
        print("Graph 6/6: Distance Evolution...")
        plot_distance_evolution(kmeans)
        
        print("\n✅ All visualizations complete!")
    
    print("\n" + "="*70)
    print("Done! Thank you for using K-Means Clustering!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
