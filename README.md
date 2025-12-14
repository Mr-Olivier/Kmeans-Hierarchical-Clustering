# 🎯 Clustering Algorithms Visualization

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-orange.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Interactive visualization of K-Means and Hierarchical clustering algorithms with step-by-step iteration tracking and comparative analysis.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Algorithms](#-algorithms-implemented)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Technologies](#-technologies)
- [Team](#-team)
- [License](#-license)

---

## 🌟 Overview

This project implements and visualizes two fundamental unsupervised learning algorithms:

1. **K-Means Clustering** - A partitioning method that divides data into K distinct clusters
2. **Hierarchical Clustering** - An agglomerative approach that builds a tree-like hierarchy of clusters

Each algorithm includes:

- ✨ **Real-time visualization** of clustering process
- 📊 **Iteration tracking** showing how clusters evolve
- 🎨 **Interactive graphs** and dendrograms
- 📈 **Performance metrics** and comparison analysis

---

## ✨ Features

### K-Means Clustering

- 🔄 Iterative centroid optimization
- 📊 Visualization of each iteration
- 🎯 Convergence analysis
- 🎨 Color-coded cluster assignments
- 📉 Within-cluster sum of squares (WCSS) tracking

### Hierarchical Clustering

- 🌲 Dendrogram visualization
- 🔗 Linkage method comparison
- 📏 Distance metrics analysis
- 🎯 Optimal cluster number detection
- 🎨 Hierarchical tree representation

### General Features

- 🖼️ High-quality matplotlib visualizations
- 💾 Save results as images
- 📊 Side-by-side algorithm comparison
- 🔧 Customizable parameters
- 📝 Detailed iteration logs

---

## 🧮 Algorithms Implemented

### 1. K-Means Clustering

**How it works:**

1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

**Key Parameters:**

- `n_clusters`: Number of clusters (default: 3)
- `max_iter`: Maximum iterations (default: 300)
- `tol`: Convergence tolerance (default: 1e-4)

**Use Cases:**

- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection

### 2. Hierarchical Clustering

**How it works:**

1. Start with each point as individual cluster
2. Merge two closest clusters
3. Update distance matrix
4. Repeat until single cluster remains

**Key Parameters:**

- `linkage`: Linkage method (ward, complete, average)
- `metric`: Distance metric (euclidean, manhattan)
- `n_clusters`: Number of final clusters

**Use Cases:**

- Gene sequence analysis
- Social network analysis
- Document organization
- Taxonomy creation

---

## 🚀 Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Mr-Olivier/Kmeans-Hierarchical-Clustering.git
cd Kmeans-Hierarchical-Clustering

# Install K-Means dependencies
pip install -r kmeans_clustering/requirements.txt

# Install Hierarchical dependencies
pip install -r hierarchical_clustering/hierarchical_requirements.txt

# Or install all at once
pip install numpy matplotlib scikit-learn scipy pandas
```

---

## 💻 Usage

### Running K-Means Clustering

```bash
# Navigate to K-Means directory
cd kmeans_clustering

# Run the algorithm
python kmeans.py
```

**Expected Output:**

- Console logs showing iteration progress
- Visualization window with clustering results
- Saved images in `output/` folder

### Running Hierarchical Clustering

```bash
# Navigate to Hierarchical directory
cd hierarchical_clustering

# Run the algorithm
python hierarchical_main.py
```

**Expected Output:**

- Dendrogram visualization
- Cluster assignments
- Distance matrix heatmap

### Custom Parameters

```python
# K-Means example
from kmeans import KMeans

kmeans = KMeans(n_clusters=5, max_iter=100)
labels = kmeans.fit(data)
kmeans.visualize()
```

```python
# Hierarchical example
from hierarchical import HierarchicalClustering

hc = HierarchicalClustering(n_clusters=4, linkage='ward')
labels = hc.fit(data)
hc.plot_dendrogram()
```

---

## 📂 Project Structure

```
Kmeans-Hierarchical-Clustering/
│
├── 📁 kmeans_clustering/
│   ├── 📄 kmeans.py                    # K-Means implementation
│   ├── 📄 visualize.py                 # Visualization utilities
│   ├── 📄 README.md                    # K-Means documentation
│   └── 📄 requirements.txt             # Dependencies
│
├── 📁 hierarchical_clustering/
│   ├── 📄 hierarchical_main.py         # Hierarchical implementation
│   ├── 📄 hierarchical.py              # Core algorithm
│   ├── 📄 HIERARCHICAL_README.md       # Documentation
│   └── 📄 hierarchical_requirements.txt
│
├── 📄 Group_Members.txt                # Team information
├── 📄 OUTPUTS_SCREENSHOOTS.docx        # Results and analysis
├── 📄 structure.txt                    # Project structure
├── 📄 .gitignore                       # Git ignore rules
└── 📄 README.md                        # This file
```

---

## 📸 Screenshots

### K-Means Clustering Results

<div align="center">

**Iteration 1**  
![K-Means Iteration 1](docs/kmeans_iter1.png)

**Iteration 5**  
![K-Means Iteration 5](docs/kmeans_iter5.png)

**Final Clustering**  
![K-Means Final](docs/kmeans_final.png)

</div>

### Hierarchical Clustering Dendrogram

<div align="center">

![Dendrogram](docs/dendrogram.png)

**Cluster Visualization**  
![Hierarchical Clusters](docs/hierarchical_clusters.png)

</div>

> 📝 _For complete results and analysis, see [OUTPUTS_SCREENSHOOTS.docx](OUTPUTS_SCREENSHOOTS.docx)_

---

## 🛠️ Technologies

### Core Libraries

| Technology                                                                              | Version | Purpose                |
| --------------------------------------------------------------------------------------- | ------- | ---------------------- |
| ![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)                    | 3.13+   | Programming language   |
| ![NumPy](https://img.shields.io/badge/NumPy-1.26-orange?logo=numpy)                     | 1.26+   | Numerical computations |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-red?logo=plotly)              | 3.8+    | Data visualization     |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-green?logo=scikit-learn) | 1.4+    | ML algorithms          |
| ![SciPy](https://img.shields.io/badge/SciPy-1.12-purple?logo=scipy)                     | 1.12+   | Scientific computing   |

### Development Tools

- **Git** - Version control
- **VS Code** - Code editor
- **Jupyter** - Interactive development
- **PowerShell** - Terminal

---

## 📊 Performance Comparison

| Metric               | K-Means        | Hierarchical          |
| -------------------- | -------------- | --------------------- |
| **Time Complexity**  | O(n·k·i·d)     | O(n²·log n)           |
| **Space Complexity** | O(n·d)         | O(n²)                 |
| **Best For**         | Large datasets | Small-medium datasets |
| **Cluster Shape**    | Spherical      | Any shape             |
| **Deterministic**    | No             | Yes                   |

**Legend:**

- `n` = number of data points
- `k` = number of clusters
- `i` = number of iterations
- `d` = number of dimensions

---

## 👥 Team

Meet our amazing team members who made this project possible:

<div align="center">

| Name         | Role                 | Contribution            |
| ------------ | -------------------- | ----------------------- |
| **Member 1** | Lead Developer       | K-Means Implementation  |
| **Member 2** | Algorithm Specialist | Hierarchical Clustering |
| **Member 3** | Visualization Lead   | Graphics & Charts       |
| **Member 4** | Documentation        | README & Reports        |

_See [Group_Members.txt](Group_Members.txt) for complete details_

</div>

---

## 📚 Learning Resources

Want to learn more about clustering? Check these out:

### 📖 Tutorials

- [K-Means Algorithm Explained](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Hierarchical Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [Unsupervised Learning Course](https://www.coursera.org/learn/machine-learning)

### 📄 Research Papers

- Lloyd, S. P. (1982). "Least squares quantization in PCM"
- Ward, J. H. (1963). "Hierarchical grouping to optimize an objective function"

### 🎥 Video Tutorials

- [StatQuest: K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [Hierarchical Clustering Explained](https://www.youtube.com/watch?v=7xHsRkOdVwo)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🔧 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

### Ideas for Contribution

- 🎨 Improve visualizations
- 📊 Add more clustering algorithms (DBSCAN, Mean Shift)
- 🔧 Optimize performance
- 📝 Enhance documentation
- 🐛 Fix bugs

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Clustering Algorithms Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🎓 Academic Context

**Course:** Artificial Intelligence  
**Topic:** Unsupervised Learning - Clustering Algorithms  
**Institution:** [Your University Name]  
**Semester:** [Current Semester]  
**Year:** 2024

---

## 🐛 Known Issues

- [ ] K-Means may not converge for very large K values
- [ ] Hierarchical clustering is memory-intensive for large datasets
- [ ] Some edge cases in visualization need improvement

**Reporting Issues:**  
Found a bug? Please [open an issue](https://github.com/Mr-Olivier/Kmeans-Hierarchical-Clustering/issues) with:

- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

---

## 🔮 Future Enhancements

- [ ] Add DBSCAN clustering algorithm
- [ ] Implement GPU acceleration
- [ ] Create web-based visualization
- [ ] Add interactive parameter tuning
- [ ] Support for 3D visualizations
- [ ] Real-time clustering on streaming data
- [ ] Export results to CSV/JSON
- [ ] Add cluster quality metrics dashboard

---

## 📞 Contact

Have questions or suggestions? Reach out!

- 📧 Email: [oiradukunda63@gmail.com]
- 🐙 GitHub: [@Mr-Olivier](https://github.com/Mr-Olivier)
- 💼 LinkedIn: [https://www.linkedin.com/in/olivier-irad/]
- 🌐 Website: [https://olivier-ira.vercel.app/]

---

## 🙏 Acknowledgments

Special thanks to:

- **Scikit-learn Team** - For the amazing ML library
- **Matplotlib Contributors** - For visualization tools
- **Our Professor** - For guidance and support
- **Stack Overflow Community** - For debugging help
- **OpenAI** - For documentation assistance

---

## ⭐ Star History

If you find this project helpful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=Mr-Olivier/Kmeans-Hierarchical-Clustering&type=Date)](https://star-history.com/#Mr-Olivier/Kmeans-Hierarchical-Clustering&Date)

---

<div align="center">

**Made with ❤️ by the Clustering Algorithms Team**

[⬆ Back to Top](#-clustering-algorithms-visualization)

---

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)

</div>
