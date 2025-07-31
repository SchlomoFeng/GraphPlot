# GraphPlot - Enhanced Graph Visualization System

An advanced graph visualization system for industrial pipeline networks with intelligent node clustering and multiple layout algorithms.

## 🚀 Key Features

### Enhanced Layout Algorithms
- **Hierarchical Layout**: Organizes nodes by type in separate layers
- **Force-Directed Layout**: Physics-based positioning with type clustering
- **Spring-Embedded Layout**: Category-based attraction/repulsion forces
- **Circular Layout**: Concentric circles by node type
- **Grid Layout**: Organized grid sections for different types
- **Layered Layout**: Hierarchical structure visualization
- **Community-Based Layout**: Automatic community detection

### Advanced Clustering
- **Node Type Clustering**: Groups similar components (Stream, Tee, Mixer, VavlePro)
- **Spatial Clustering**: Proximity-based grouping with DBSCAN
- **Community Detection**: Network structure-based clustering
- **Hybrid Clustering**: Combines multiple strategies

### Visual Enhancements
- **Color-coded node types** for easy identification
- **Collision detection** to prevent overlapping
- **Configurable spacing** between nodes and clusters
- **Professional legends** and labels
- **High-quality PNG output**

## 📦 Installation

```bash
pip install matplotlib networkx pandas scikit-learn numpy python-louvain
```

## 🎯 Quick Start

```python
from enhanced_graph_plot import EnhancedGraphPlotter, LayoutConfig, LayoutType, ClusteringStrategy

# Create configuration
config = LayoutConfig(
    layout_type=LayoutType.HIERARCHICAL,
    clustering_strategy=ClusteringStrategy.NODE_TYPE,
    node_spacing=80.0,
    cluster_spacing=300.0
)

# Create and run plotter
plotter = EnhancedGraphPlotter(config)
plotter.load_data('your_data.json')
plotter.visualize(save_path='enhanced_plot.png')
```

## 🎨 Output Examples

The system generates multiple visualization styles:
- `enhanced_graph_plot.png` - Main enhanced visualization
- `layout_comparison.png` - Layout algorithm comparison
- `clustering_*.png` - Different clustering approaches
- `before_after_comparison.png` - Improvement demonstration

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_enhanced_system.py
```

## 📊 Performance

- Processes 200+ nodes in ~0.04 seconds
- Supports 4+ node types automatically
- 7 layout algorithms available
- 4 clustering strategies implemented

## 🔧 Configuration Options

The `LayoutConfig` class provides extensive customization:
- Layout algorithm selection
- Clustering strategy choice
- Spacing parameters
- Visual styling options
- Collision detection settings

## 📈 Improvements Over Original

✅ **Organized node types** instead of mixed clusters  
✅ **Multiple layout algorithms** vs single approach  
✅ **Automatic type detection** from JSON data  
✅ **Collision detection** prevents overlapping  
✅ **Professional visualization** with legends  
✅ **Comprehensive configuration** system