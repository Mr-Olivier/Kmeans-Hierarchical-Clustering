# K-Means Clustering Algorithm

**Author:** Your Name  
**Course:** Artificial Intelligence / Machine Learning  
**Date:** December 2025

## Overview

This project implements the K-Means clustering algorithm with comprehensive visualization capabilities. The implementation includes:

- ✅ Core K-means algorithm with step-by-step tracking
- ✅ Multiple visualization methods
- ✅ Interactive user input
- ✅ Detailed iteration logging
- ✅ Professional plots and graphs

## Project Structure

```
kmeans_clustering/
│
├── kmeans.py           # Core K-means implementation            # Main program entry point
├── requirements.txt    # Python dependencies
├── README.md          # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Install Python

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/)

### Step 2: Clone or Download Project

Download the project files to your computer and navigate to the folder:

```bash
cd kmeans_clustering
```

### Step 3: Install Dependencies

Run one of these commands:

**Option A: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Option B: Manual installation**

```bash
pip install numpy matplotlib
```

**For Mac/Linux, you might need:**

```bash
pip3 install -r requirements.txt
```

## How to Run

### Method 1: Run the Main Program

```bash
python kmeans.py
```

### Method 2: In VS Code

1. Open the folder in VS Code
2. Open `main.py`
3. Press `F5` or click "Run" → "Run Without Debugging"
4. Or click the ▶️ play button in the top-right corner

### Method 3: Python Interactive Mode

```bash
python
>>> from main import main
>>> main()
```

## Usage Examples

### Example 1: Use Built-in Example Data

```
Choose an option:
  1. Use example data from assignment
  2. Enter custom data

Your choice (1 or 2): 1

Would you like to see visualizations? (y/n): y
```

### Example 2: Enter Custom Data

```
Choose an option:
  1. Use example data from assignment
  2. Enter custom data

Your choice (1 or 2): 2

How many points do you want to cluster? 5

Enter 5 points in format: name x y
Point 1: P1 1 1
Point 2: P2 2 1
Point 3: P3 1 2
Point 4: P4 10 10
Point 5: P5 11 10

How many clusters (1-5)? 2

Enter 2 indices (0-4) for initial centers
Initial center indices: 0 3

How many iterations? (recommended: 2-5): 3

Would you like to see visualizations? (y/n): y
```

## Input Format

### Points

Enter points in the format: `name x y`

- **name**: Identifier (e.g., A1, P1, Point1)
- **x**: X-coordinate (number)
- **y**: Y-coordinate (number)

### Initial Centers

Enter the indices (0-based) of points to use as initial centroids.

- For 8 points, valid indices are 0-7
- Example: `0 3 6` uses points 0, 3, and 6

## Output

### Console Output

The program prints:

- Iteration details with centroids
- Cluster assignments
- Distances from points to centroids
- Final clustering summary

## Example Output

### Assignment Example

Using the example data:

- Points: A1(2,10), A2(2,5), A3(8,4), A4(5,8), A5(7,5), A6(6,4), A7(1,2), A8(4,9)
- Initial centers: A1, A4, A7
- Clusters: 3
- Iterations: 2

**Final Centers (after 2 iterations):**

- C1: (3.67, 9.00)
- C2: (7.00, 4.33)
- C3: (1.50, 3.50)

## Understanding the Visualizations

### Color Coding

- **Red, Blue, Green** - Different clusters
- **Black X with yellow border** - Centroids
- **Dashed lines** - Distance from point to assigned centroid
