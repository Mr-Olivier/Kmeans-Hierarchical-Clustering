
"""
Hierarchical Clustering - Complete Implementation in One File
Author: Your Name
Course: AI/Machine Learning Assignment
Agglomerative (Bottom-up) approach
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage


class HierarchicalClustering:
    """Hierarchical Clustering using Agglomerative approach"""
    
    def __init__(self, linkage='single'):
        self.linkage = linkage
        self.clusters = []
        self.merge_history = []
        self.distance_matrices = []
        
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def calculate_distance_matrix(self, X):
        """Calculate pairwise distance matrix"""
        n = len(X)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.euclidean_distance(X[i], X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def cluster_distance(self, cluster1, cluster2, X):
        """Calculate distance between two clusters"""
        distances = []
        
        for i in cluster1:
            for j in cluster2:
                dist = self.euclidean_distance(X[i], X[j])
                distances.append(dist)
        
        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return np.mean(distances)
        else:
            return min(distances)
    
    def fit(self, X, point_names=None):
        """Perform hierarchical clustering"""
        n = len(X)
        
        self.clusters = [[i] for i in range(n)]
        
        if point_names is None:
            point_names = [f"P{i+1}" for i in range(n)]
        
        self.point_names = point_names
        self.X = X
        
        initial_matrix = self.calculate_distance_matrix(X)
        self.distance_matrices.append({
            'iteration': 0,
            'matrix': initial_matrix.copy(),
            'clusters': [cluster.copy() for cluster in self.clusters]
        })
        
        iteration = 1
        
        while len(self.clusters) > 1:
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    dist = self.cluster_distance(self.clusters[i], 
                                                 self.clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            cluster_i = self.clusters[merge_i]
            cluster_j = self.clusters[merge_j]
            new_cluster = cluster_i + cluster_j
            
            self.merge_history.append({
                'iteration': iteration,
                'cluster1': cluster_i.copy(),
                'cluster2': cluster_j.copy(),
                'merged': new_cluster.copy(),
                'distance': min_dist
            })
            
            self.clusters.pop(max(merge_i, merge_j))
            self.clusters.pop(min(merge_i, merge_j))
            self.clusters.append(new_cluster)
            
            current_matrix = self.calculate_cluster_matrix(X)
            self.distance_matrices.append({
                'iteration': iteration,
                'matrix': current_matrix,
                'clusters': [cluster.copy() for cluster in self.clusters]
            })
            
            iteration += 1
        
        return self
    
    def calculate_cluster_matrix(self, X):
        """Calculate distance matrix between current clusters"""
        n = len(self.clusters)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.cluster_distance(self.clusters[i], 
                                             self.clusters[j], X)
                matrix[i, j] = dist
                matrix[j, i] = dist
        
        return matrix
    
    def get_cluster_names(self, cluster_indices):
        """Get names of points in a cluster"""
        names = [self.point_names[i] for i in sorted(cluster_indices)]
        if len(names) == 1:
            return names[0]
        else:
            return '{' + ','.join(names) + '}'
    
    def print_distance_matrix(self, iteration):
        """Print the distance matrix for a specific iteration"""
        data = self.distance_matrices[iteration]
        matrix = data['matrix']
        clusters = data['clusters']
        
        print(f"\n{'='*70}")
        print(f"Distance Matrix - Iteration {iteration}")
        print(f"{'='*70}")
        
        labels = [self.get_cluster_names(cluster) for cluster in clusters]
        
        print(f"\n{'':15s}", end='')
        for label in labels:
            print(f"{label:12s}", end='')
        print()
        
        for i, label in enumerate(labels):
            print(f"{label:15s}", end='')
            for j in range(len(labels)):
                if i == j:
                    print(f"{'0':12s}", end='')
                else:
                    print(f"{matrix[i, j]:12.2f}", end='')
            print()
    
    def print_iteration_details(self, iteration):
        """Print detailed information about an iteration"""
        if iteration == 0:
            print("\n" + "="*70)
            print("INITIAL STATE")
            print("="*70)
            self.print_distance_matrix(0)
            return
        
        if iteration > len(self.merge_history):
            print(f"No data for iteration {iteration}")
            return
        
        merge = self.merge_history[iteration - 1]
        
        print("\n" + "="*70)
        print(f"ITERATION {iteration}")
        print("="*70)
        
        cluster1_name = self.get_cluster_names(merge['cluster1'])
        cluster2_name = self.get_cluster_names(merge['cluster2'])
        merged_name = self.get_cluster_names(merge['merged'])
        
        print(f"\nMinimum distance = {merge['distance']:.2f}")
        print(f"Merging: {cluster1_name} ∪ {cluster2_name} → {merged_name}")
        
        self.print_distance_matrix(iteration)
    
    def print_summary(self):
        """Print summary of clustering process"""
        print("\n" + "="*70)
        print("HIERARCHICAL CLUSTERING SUMMARY")
        print("="*70)
        
        print(f"\nLinkage method: {self.linkage}")
        print(f"Number of points: {len(self.X)}")
        print(f"Number of merges: {len(self.merge_history)}")
        
        print("\nMerge sequence:")
        for i, merge in enumerate(self.merge_history, 1):
            cluster1_name = self.get_cluster_names(merge['cluster1'])
            cluster2_name = self.get_cluster_names(merge['cluster2'])
            merged_name = self.get_cluster_names(merge['merged'])
            print(f"  {i}. {cluster1_name} + {cluster2_name} → "
                  f"{merged_name} (distance: {merge['distance']:.2f})")
    
    def get_clusters_at_distance(self, threshold):
        """Get clusters at a specific distance threshold"""
        clusters = [[i] for i in range(len(self.X))]
        
        for merge in self.merge_history:
            if merge['distance'] <= threshold:
                idx1 = None
                idx2 = None
                
                for i, cluster in enumerate(clusters):
                    if merge['cluster1'][0] in cluster:
                        idx1 = i
                    if merge['cluster2'][0] in cluster:
                        idx2 = i
                
                if idx1 is not None and idx2 is not None and idx1 != idx2:
                    new_cluster = clusters[idx1] + clusters[idx2]
                    clusters.pop(max(idx1, idx2))
                    clusters.pop(min(idx1, idx2))
                    clusters.append(new_cluster)
        
        return clusters
    
    def print_clustering_at_thresholds(self):
        """Print clustering results at different thresholds"""
        print("\n" + "="*70)
        print("CLUSTERING AT DIFFERENT THRESHOLDS")
        print("="*70)
        
        distances = sorted(set([merge['distance'] for merge in self.merge_history]))
        
        for dist in distances:
            clusters = self.get_clusters_at_distance(dist - 0.01)
            n_clusters = len(clusters)
            
            print(f"\nHeight < {dist:.2f}: {n_clusters} clusters")
            for cluster in clusters:
                cluster_name = self.get_cluster_names(cluster)
                print(f"  {cluster_name}")


def calculate_all_distances(X, point_names):
    """Calculate and display all pairwise distances"""
    print("\n" + "="*70)
    print("INITIAL DISTANCE CALCULATIONS")
    print("="*70)
    
    n = len(X)
    
    print("\nDetailed calculations:")
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = X[i]
            x2, y2 = X[j]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            print(f"d({point_names[i]},{point_names[j]}) = "
                  f"√[({x2}-{x1})² + ({y2}-{y1})²] = "
                  f"√[{(x2-x1)**2:.2f} + {(y2-y1)**2:.2f}] = {dist:.2f}")


def plot_dendrogram(hc, point_names):
    """Plot dendrogram showing hierarchical clustering"""
    n = len(hc.X)
    Z = []
    
    cluster_map = {i: i for i in range(n)}
    next_cluster_id = n
    
    for merge in hc.merge_history:
        c1_id = cluster_map[merge['cluster1'][0]]
        c2_id = cluster_map[merge['cluster2'][0]]
        
        Z.append([c1_id, c2_id, merge['distance'], len(merge['merged'])])
        
        for idx in merge['merged']:
            cluster_map[idx] = next_cluster_id
        
        next_cluster_id += 1
    
    Z = np.array(Z)
    
    plt.figure(figsize=(12, 8))
    
    dendro = dendrogram(Z, labels=point_names, 
                       leaf_font_size=12,
                       color_threshold=0)
    
    plt.xlabel('Data Points', fontsize=12, fontweight='bold')
    plt.ylabel('Distance', fontsize=12, fontweight='bold')
    plt.title('Hierarchical Clustering Dendrogram', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_clusters_scatter(X, point_names, hc, threshold=None):
    """Plot scatter plot of points with clusters"""
    plt.figure(figsize=(10, 8))
    
    if threshold is not None:
        clusters = hc.get_clusters_at_distance(threshold)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, cluster in enumerate(clusters):
            cluster_points = X[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=colors[i % len(colors)], s=200, alpha=0.6,
                       edgecolors='black', linewidth=2,
                       label=f'Cluster {i+1}')
    else:
        plt.scatter(X[:, 0], X[:, 1], c='blue', s=200, alpha=0.6,
                   edgecolors='black', linewidth=2)
    
    for i, (point, name) in enumerate(zip(X, point_names)):
        plt.annotate(name, xy=(point[0], point[1]),
                    xytext=(-8, -8), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    plt.xlabel('X Coordinate', fontsize=12, fontweight='bold')
    plt.ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    
    if threshold:
        plt.title(f'Clusters at Distance Threshold = {threshold:.2f}',
                 fontsize=14, fontweight='bold')
        plt.legend()
    else:
        plt.title('Data Points', fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main program"""
    
    print("\n" + "="*70)
    print("HIERARCHICAL CLUSTERING")
    print("="*70 + "\n")
    
    print("1. Use example data from class")
    print("2. Enter custom data")
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == '1':
        points = np.array([
            [1, 1],      # A
            [1.5, 1.5],  # B
            [5, 5],      # C
            [3, 4],      # D
            [4, 4],      # E
            [3, 3.5]     # F
        ])
        point_names = ['A', 'B', 'C', 'D', 'E', 'F']
        
        print("\nUsing example data from class:")
        print("Points: A(1,1), B(1.5,1.5), C(5,5)")
        print("        D(3,4), E(4,4), F(3,3.5)")
        
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
    
    print("\nLinkage methods:")
    print("1. Single linkage (minimum distance)")
    print("2. Complete linkage (maximum distance)")
    print("3. Average linkage (average distance)")
    
    linkage_choice = input("\nChoice (1/2/3, default=1): ").strip() or '1'
    
    linkage_map = {'1': 'single', '2': 'complete', '3': 'average'}
    linkage = linkage_map.get(linkage_choice, 'single')
    
    calculate_all_distances(points, point_names)
    
    print("\n" + "="*70)
    print("Running hierarchical clustering...")
    print("="*70)
    
    hc = HierarchicalClustering(linkage=linkage)
    hc.fit(points, point_names)
    
    for i in range(len(hc.merge_history) + 1):
        hc.print_iteration_details(i)
    
    hc.print_summary()
    
    hc.print_clustering_at_thresholds()
    
    print("\n" + "="*70)
    show_viz = input("\nShow visualizations? (y/n): ").strip().lower()
    
    if show_viz == 'y':
        print("\nGenerating visualizations...")
        
        try:
            print("\n1. Showing dendrogram...")
            plot_dendrogram(hc, point_names)
        except ImportError:
            print("\n⚠️  scipy not installed. Skipping dendrogram.")
            print("   Install with: pip install scipy")
        except:
            print("\n⚠️  Could not create dendrogram.")
        
        print("\n2. Showing data points...")
        plot_clusters_scatter(points, point_names, hc)
        
        show_threshold = input("\nShow clusters at specific threshold? (y/n): ").strip().lower()
        if show_threshold == 'y':
            threshold = float(input("Enter distance threshold: "))
            plot_clusters_scatter(points, point_names, hc, threshold)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
