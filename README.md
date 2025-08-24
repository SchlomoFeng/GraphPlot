# GraphPlot

Enhanced graph visualization tool with node type differentiation using marker shapes.

## Features

- **Multi-Algorithm Clustering**: DBSCAN, Hierarchical, and K-means clustering
- **Node Type Visualization**: Different marker shapes for different node types
- **Dual Legend System**: Shows both cluster colors and node type markers
- **Export Functionality**: Saves visualizations as PNG images

## Node Type Markers

The visualization uses different marker shapes to distinguish node types:

- **Stream**: Circle (○) - `marker='o'`
- **Mixer**: Square (■) - `marker='s'`
- **ValvePro**: Triangle (▲) - `marker='^'`
- **Tee**: Diamond (◆) - `marker='D'`
- **Unknown**: Pentagon (⬟) - `marker='p'` (default for unrecognized types)

## Usage

```python
from GraphPlot_0708YTS4 import main

# Run with debug mode to use local data file
main(debug=True)
```

Or run the demo script:

```bash
python3 demo_markers.py
```

## Output

The tool generates three visualization files:
- `DBSCAN聚类结果_with_markers.png`
- `层次聚类结果_with_markers.png` 
- `K-means聚类结果_with_markers.png`

Each visualization shows nodes colored by cluster membership and shaped by node type, with dual legends for easy interpretation.