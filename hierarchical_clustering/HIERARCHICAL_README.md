# Hierarchical Clustering Assignment

Implementation of Agglomerative Hierarchical Clustering algorithm.

## Files

- `hierarchical_main.py` - Main program (run this)
- `hierarchical_requirements.txt` - Required packages

## How to Run

1. Install required packages:

```bash
pip install numpy matplotlib scipy
```

2. Run the program:

```bash
python hierarchical_main.py
```

3. Choose option 1 for example data (from class) or option 2 for custom data

4. When asked, type 'y' to see the visualizations

## What it Does

The program:

- Implements agglomerative hierarchical clustering
- Shows step-by-step merge iterations
- Displays distance matrices at each step
- Creates dendrogram visualization
- Shows clusters at different thresholds

## Example Data (From Class)

Points:

- A: (1, 1)
- B: (1.5, 1.5)
- C: (5, 5)
- D: (3, 4)
- E: (4, 4)
- F: (3, 3.5)

## Algorithm Steps

1. Start: Each point is its own cluster
2. Calculate distances between all pairs
3. Find closest pair of clusters
4. Merge them into one cluster
5. Update distance matrix
6. Repeat steps 3-5 until one cluster remains

## Linkage Methods

- **Single linkage**: Minimum distance between clusters
- **Complete linkage**: Maximum distance between clusters
- **Average linkage**: Average distance between clusters

## Output

### Console:

- Initial distance calculations
- Distance matrix at each iteration
- Merge sequence with distances
- Clusters at different thresholds

### Visualizations:

1. **Dendrogram**: Tree showing merge hierarchy
2. **Scatter plot**: Points visualization
3. **Cluster plot**: Clusters at specific threshold

## Example Output

```
ITERATION 1
Minimum distance = 0.50
Merging: D ∪ F → {D,F}

Distance Matrix:
         A        B        C        E     {D,F}
A        0     0.71     5.66     4.24     3.20
B              0     4.95     3.54     2.50
C                       0     1.41     2.24
E                                0     1.00
{D,F}                                      0
```

That's it!
