"""
Enhanced Graph Visualization System
Addresses the clustering and layout issues identified in the original GraphPlot system.

Key improvements:
1. Node type-based clustering
2. Multiple layout algorithms (hierarchical, force-directed, spring-embedded)
3. Better visual separation of different node types
4. Configurable spacing and positioning
5. Community detection and collision avoidance
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import community  # python-louvain for community detection
from scipy.spatial.distance import pdist, squareform
from matplotlib.patches import Rectangle
import itertools
from collections import defaultdict


class LayoutType(Enum):
    """Available layout algorithms"""
    HIERARCHICAL = "hierarchical"
    FORCE_DIRECTED = "force_directed"
    SPRING_EMBEDDED = "spring_embedded"
    CIRCULAR = "circular"
    GRID = "grid"
    LAYERED = "layered"
    COMMUNITY_BASED = "community_based"


class ClusteringStrategy(Enum):
    """Available clustering strategies"""
    NODE_TYPE = "node_type"
    SPATIAL = "spatial"
    COMMUNITY = "community"
    HYBRID = "hybrid"


@dataclass
class LayoutConfig:
    """Configuration for layout algorithms and visual settings"""
    layout_type: LayoutType = LayoutType.FORCE_DIRECTED
    clustering_strategy: ClusteringStrategy = ClusteringStrategy.NODE_TYPE
    
    # Spacing parameters
    node_spacing: float = 50.0
    cluster_spacing: float = 200.0
    edge_spacing: float = 10.0
    
    # Force-directed parameters
    k_spring: float = 1.0
    iterations: int = 50
    
    # Visual parameters
    node_size_base: float = 300
    node_size_multiplier: float = 1.5
    edge_width_base: float = 1.0
    edge_alpha: float = 0.6
    
    # Collision detection
    enable_collision_detection: bool = True
    collision_margin: float = 20.0
    
    # Color schemes for different node types
    node_type_colors: Dict[str, str] = field(default_factory=lambda: {
        'Stream': '#FF6B6B',      # Red for streams
        'Tee': '#4ECDC4',         # Teal for tees
        'Mixer': '#45B7D1',       # Blue for mixers
        'VavlePro': '#96CEB4',    # Green for valves
        'default': '#DDA0DD'      # Purple for others
    })


class EnhancedGraphPlotter:
    """Enhanced graph plotter with advanced layout and clustering capabilities"""
    
    def __init__(self, config: LayoutConfig = None):
        self.config = config or LayoutConfig()
        self.node_df: Optional[pd.DataFrame] = None
        self.edge_df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.node_types: Dict[str, str] = {}
        self.node_positions: Dict[str, Tuple[float, float]] = {}
        self.node_clusters: Dict[str, int] = {}
        
    def load_data(self, json_file_path: str) -> bool:
        """Load graph data from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Extract nodes and edges
            nodelist = data['nodelist'] 
            edgelist = data['linklist']
            
            self.node_df = pd.DataFrame(nodelist)
            self.edge_df = pd.DataFrame(edgelist)
            
            print(f"Loaded {len(self.node_df)} nodes and {len(self.edge_df)} edges")
            
            # Extract node types and positions
            self._extract_node_info()
            
            # Build graph
            self._build_graph()
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _extract_node_info(self):
        """Extract node types and positions from the data"""
        self.node_types = {}
        initial_positions = {}
        
        for i in range(len(self.node_df)):
            node_id = self.node_df['id'].iloc[i]
            
            # Extract node type from parameter
            try:
                if isinstance(self.node_df['parameter'].iloc[i], str):
                    param = json.loads(self.node_df['parameter'].iloc[i])
                else:
                    param = self.node_df['parameter'].iloc[i]
                
                # Get node type
                node_type = param.get('type', 'Unknown')
                self.node_types[node_id] = node_type
                
                # Get position if available
                if 'styles' in param and 'position' in param['styles']:
                    pos = param['styles']['position']
                    initial_positions[node_id] = (pos['x'], pos['y'])
                else:
                    # Default position
                    initial_positions[node_id] = (0, 0)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse node {node_id}: {e}")
                self.node_types[node_id] = 'Unknown'
                initial_positions[node_id] = (0, 0)
        
        self.node_positions = initial_positions
        print(f"Extracted {len(set(self.node_types.values()))} unique node types:")
        type_counts = pd.Series(list(self.node_types.values())).value_counts()
        for node_type, count in type_counts.items():
            print(f"  {node_type}: {count}")
    
    def _build_graph(self):
        """Build NetworkX graph from the data"""
        self.graph = nx.Graph()
        
        # Add nodes with attributes
        for node_id in self.node_types.keys():
            self.graph.add_node(node_id, 
                              node_type=self.node_types[node_id],
                              pos=self.node_positions[node_id])
        
        # Add edges
        valid_edges = 0
        for i in range(len(self.edge_df)):
            source_id = self.edge_df['sourceid'].iloc[i]
            target_id = self.edge_df['targetid'].iloc[i]
            
            if source_id in self.node_types and target_id in self.node_types:
                # Extract edge weight/length
                try:
                    if isinstance(self.edge_df['parameter'].iloc[i], str):
                        edge_param = json.loads(self.edge_df['parameter'].iloc[i])
                    else:
                        edge_param = self.edge_df['parameter'].iloc[i]
                    
                    length = edge_param.get('parameter', {}).get('Length', 1.0)
                    if length is None or length <= 0:
                        length = 1.0
                        
                    self.graph.add_edge(source_id, target_id, weight=length)
                    valid_edges += 1
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    self.graph.add_edge(source_id, target_id, weight=1.0)
                    valid_edges += 1
        
        print(f"Built graph with {len(self.graph.nodes())} nodes and {valid_edges} edges")
        
        # Handle disconnected components
        if not nx.is_connected(self.graph):
            print("Warning: Graph is not connected, using largest connected component")
            largest_cc = max(nx.connected_components(self.graph), key=len)
            self.graph = self.graph.subgraph(largest_cc).copy()
            print(f"Using component with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
    
    def apply_clustering(self) -> Dict[str, int]:
        """Apply clustering based on the selected strategy"""
        if self.config.clustering_strategy == ClusteringStrategy.NODE_TYPE:
            return self._cluster_by_node_type()
        elif self.config.clustering_strategy == ClusteringStrategy.SPATIAL:
            return self._cluster_by_spatial_proximity()
        elif self.config.clustering_strategy == ClusteringStrategy.COMMUNITY:
            return self._cluster_by_community_detection()
        elif self.config.clustering_strategy == ClusteringStrategy.HYBRID:
            return self._cluster_hybrid()
        else:
            return self._cluster_by_node_type()
    
    def _cluster_by_node_type(self) -> Dict[str, int]:
        """Cluster nodes by their type"""
        type_to_cluster = {}
        cluster_id = 0
        
        for node_type in set(self.node_types.values()):
            type_to_cluster[node_type] = cluster_id
            cluster_id += 1
        
        clusters = {}
        for node_id, node_type in self.node_types.items():
            if node_id in self.graph.nodes():
                clusters[node_id] = type_to_cluster[node_type]
        
        self.node_clusters = clusters
        print(f"Created {len(type_to_cluster)} clusters by node type")
        return clusters
    
    def _cluster_by_spatial_proximity(self) -> Dict[str, int]:
        """Cluster nodes by spatial proximity using DBSCAN"""
        nodes = list(self.graph.nodes())
        positions = np.array([self.node_positions[node] for node in nodes])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.config.cluster_spacing, min_samples=2)
        cluster_labels = clustering.fit_predict(positions)
        
        clusters = {}
        for i, node in enumerate(nodes):
            clusters[node] = max(0, cluster_labels[i])  # Convert -1 (noise) to 0
        
        self.node_clusters = clusters
        print(f"Created {len(set(cluster_labels))} spatial clusters")
        return clusters
    
    def _cluster_by_community_detection(self) -> Dict[str, int]:
        """Cluster nodes using community detection algorithms"""
        try:
            # Use Louvain algorithm for community detection
            partition = community.best_partition(self.graph)
            self.node_clusters = partition
            print(f"Created {len(set(partition.values()))} communities")
            return partition
        except ImportError:
            print("Warning: python-louvain not available, falling back to node type clustering")
            return self._cluster_by_node_type()
    
    def _cluster_hybrid(self) -> Dict[str, int]:
        """Hybrid clustering combining node type and spatial/community information"""
        # First cluster by node type
        type_clusters = self._cluster_by_node_type()
        
        # Then apply community detection within each type
        hybrid_clusters = {}
        cluster_offset = 0
        
        for node_type in set(self.node_types.values()):
            # Get nodes of this type
            type_nodes = [node for node, ntype in self.node_types.items() 
                         if ntype == node_type and node in self.graph.nodes()]
            
            if len(type_nodes) <= 1:
                # Single node or empty, assign to base cluster
                for node in type_nodes:
                    hybrid_clusters[node] = cluster_offset
                cluster_offset += 1
                continue
            
            # Create subgraph for this node type
            subgraph = self.graph.subgraph(type_nodes)
            
            if len(subgraph.edges()) == 0:
                # No edges within type, use spatial clustering
                positions = np.array([self.node_positions[node] for node in type_nodes])
                if len(positions) > 1:
                    clustering = AgglomerativeClustering(n_clusters=min(3, len(type_nodes)))
                    sub_labels = clustering.fit_predict(positions)
                else:
                    sub_labels = [0]
            else:
                # Use community detection within type
                try:
                    partition = community.best_partition(subgraph)
                    sub_labels = [partition[node] for node in type_nodes]
                except:
                    sub_labels = [0] * len(type_nodes)
            
            # Assign clusters with offset
            for i, node in enumerate(type_nodes):
                hybrid_clusters[node] = cluster_offset + sub_labels[i]
            
            cluster_offset += max(sub_labels) + 1
        
        self.node_clusters = hybrid_clusters
        print(f"Created {len(set(hybrid_clusters.values()))} hybrid clusters")
        return hybrid_clusters
    
    def apply_layout(self) -> Dict[str, Tuple[float, float]]:
        """Apply the selected layout algorithm"""
        if self.config.layout_type == LayoutType.HIERARCHICAL:
            return self._layout_hierarchical()
        elif self.config.layout_type == LayoutType.FORCE_DIRECTED:
            return self._layout_force_directed()
        elif self.config.layout_type == LayoutType.SPRING_EMBEDDED:
            return self._layout_spring_embedded()
        elif self.config.layout_type == LayoutType.CIRCULAR:
            return self._layout_circular()
        elif self.config.layout_type == LayoutType.GRID:
            return self._layout_grid()
        elif self.config.layout_type == LayoutType.LAYERED:
            return self._layout_layered()
        elif self.config.layout_type == LayoutType.COMMUNITY_BASED:
            return self._layout_community_based()
        else:
            return self._layout_force_directed()
    
    def _layout_hierarchical(self) -> Dict[str, Tuple[float, float]]:
        """Hierarchical layout with node types in separate layers"""
        positions = {}
        
        # Group nodes by type
        type_groups = defaultdict(list)
        for node, node_type in self.node_types.items():
            if node in self.graph.nodes():
                type_groups[node_type].append(node)
        
        y_offset = 0
        layer_height = self.config.cluster_spacing * 2
        
        for node_type, nodes in type_groups.items():
            # Arrange nodes in this layer horizontally
            if len(nodes) == 1:
                positions[nodes[0]] = (0, y_offset)
            else:
                total_width = (len(nodes) - 1) * self.config.node_spacing
                start_x = -total_width / 2
                
                for i, node in enumerate(nodes):
                    x = start_x + i * self.config.node_spacing
                    positions[node] = (x, y_offset)
            
            y_offset += layer_height
        
        self.node_positions = positions
        return positions
    
    def _layout_force_directed(self) -> Dict[str, Tuple[float, float]]:
        """Force-directed layout with cluster-based modifications"""
        # Start with NetworkX spring layout
        pos = nx.spring_layout(self.graph, 
                              k=self.config.k_spring,
                              iterations=self.config.iterations)
        
        # Convert to our coordinate system and scale
        scale_factor = self.config.cluster_spacing * 5
        positions = {}
        for node, (x, y) in pos.items():
            positions[node] = (x * scale_factor, y * scale_factor)
        
        # Apply cluster-based adjustments
        if hasattr(self, 'node_clusters') and self.node_clusters:
            positions = self._adjust_positions_for_clusters(positions)
        
        # Apply collision detection if enabled
        if self.config.enable_collision_detection:
            positions = self._apply_collision_detection(positions)
        
        self.node_positions = positions
        return positions
    
    def _layout_spring_embedded(self) -> Dict[str, Tuple[float, float]]:
        """Spring-embedded layout with category-based attraction/repulsion"""
        positions = {}
        nodes = list(self.graph.nodes())
        
        # Initialize positions randomly
        np.random.seed(42)  # For reproducibility
        for node in nodes:
            positions[node] = (np.random.uniform(-100, 100), np.random.uniform(-100, 100))
        
        # Iterative spring simulation
        for iteration in range(self.config.iterations):
            forces = {node: [0, 0] for node in nodes}
            
            # Repulsive forces between all nodes
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    pos1 = np.array(positions[node1])
                    pos2 = np.array(positions[node2])
                    diff = pos1 - pos2
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0:
                        # Base repulsion
                        repulsion_strength = 1000.0 / (dist ** 2)
                        
                        # Modify based on node types
                        type1 = self.node_types.get(node1, 'Unknown')
                        type2 = self.node_types.get(node2, 'Unknown')
                        
                        if type1 == type2:
                            # Same type: weaker repulsion (allow clustering)
                            repulsion_strength *= 0.5
                        else:
                            # Different type: stronger repulsion (separate clusters)
                            repulsion_strength *= 2.0
                        
                        force = diff / dist * repulsion_strength
                        forces[node1][0] += force[0]
                        forces[node1][1] += force[1]
                        forces[node2][0] -= force[0]
                        forces[node2][1] -= force[1]
            
            # Attractive forces for connected nodes
            for edge in self.graph.edges():
                node1, node2 = edge
                pos1 = np.array(positions[node1])
                pos2 = np.array(positions[node2])
                diff = pos2 - pos1
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    # Spring attraction
                    ideal_length = self.config.node_spacing
                    attraction_strength = self.config.k_spring * (dist - ideal_length)
                    
                    force = diff / dist * attraction_strength
                    forces[node1][0] += force[0]
                    forces[node1][1] += force[1]
                    forces[node2][0] -= force[0]
                    forces[node2][1] -= force[1]
            
            # Update positions
            damping = 0.9
            for node in nodes:
                positions[node] = (
                    positions[node][0] + forces[node][0] * 0.01 * damping,
                    positions[node][1] + forces[node][1] * 0.01 * damping
                )
        
        self.node_positions = positions
        return positions
    
    def _layout_circular(self) -> Dict[str, Tuple[float, float]]:
        """Circular layout with node types in concentric circles"""
        positions = {}
        
        # Group nodes by type
        type_groups = defaultdict(list)
        for node, node_type in self.node_types.items():
            if node in self.graph.nodes():
                type_groups[node_type].append(node)
        
        radius_base = self.config.cluster_spacing
        type_list = list(type_groups.keys())
        
        for i, (node_type, nodes) in enumerate(type_groups.items()):
            radius = radius_base * (i + 1)
            angle_step = 2 * np.pi / len(nodes) if len(nodes) > 1 else 0
            
            for j, node in enumerate(nodes):
                angle = j * angle_step
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions[node] = (x, y)
        
        self.node_positions = positions
        return positions
    
    def _layout_grid(self) -> Dict[str, Tuple[float, float]]:
        """Grid layout with node types in different sections"""
        positions = {}
        
        # Group nodes by type
        type_groups = defaultdict(list)
        for node, node_type in self.node_types.items():
            if node in self.graph.nodes():
                type_groups[node_type].append(node)
        
        # Calculate grid dimensions
        num_types = len(type_groups)
        grid_cols = int(np.ceil(np.sqrt(num_types)))
        grid_rows = int(np.ceil(num_types / grid_cols))
        
        type_idx = 0
        for node_type, nodes in type_groups.items():
            # Calculate section position
            section_row = type_idx // grid_cols
            section_col = type_idx % grid_cols
            
            section_x = section_col * self.config.cluster_spacing * 3
            section_y = section_row * self.config.cluster_spacing * 3
            
            # Arrange nodes within section
            nodes_per_row = int(np.ceil(np.sqrt(len(nodes))))
            for i, node in enumerate(nodes):
                row = i // nodes_per_row
                col = i % nodes_per_row
                
                x = section_x + col * self.config.node_spacing
                y = section_y + row * self.config.node_spacing
                positions[node] = (x, y)
            
            type_idx += 1
        
        self.node_positions = positions
        return positions
    
    def _layout_layered(self) -> Dict[str, Tuple[float, float]]:
        """Layered layout for hierarchical structures"""
        positions = {}
        
        # Try to detect hierarchy using node degrees and types
        node_layers = {}
        
        # Classify nodes by type and connectivity
        streams = [n for n in self.graph.nodes() if self.node_types.get(n, '') == 'Stream']
        mixers = [n for n in self.graph.nodes() if self.node_types.get(n, '') == 'Mixer']
        others = [n for n in self.graph.nodes() if n not in streams and n not in mixers]
        
        # Layer assignment: streams at top, mixers in middle, others at bottom
        layer_0 = streams
        layer_1 = mixers
        layer_2 = others
        
        layers = [layer_0, layer_1, layer_2]
        
        layer_y = 0
        for layer in layers:
            if not layer:
                continue
                
            # Arrange nodes in layer horizontally
            if len(layer) == 1:
                positions[layer[0]] = (0, layer_y)
            else:
                total_width = (len(layer) - 1) * self.config.node_spacing
                start_x = -total_width / 2
                
                for i, node in enumerate(layer):
                    x = start_x + i * self.config.node_spacing
                    positions[node] = (x, layer_y)
            
            layer_y -= self.config.cluster_spacing * 2
        
        self.node_positions = positions
        return positions
    
    def _layout_community_based(self) -> Dict[str, Tuple[float, float]]:
        """Layout based on detected communities"""
        # Apply community detection first
        clusters = self.apply_clustering()
        
        positions = {}
        cluster_centers = {}
        
        # Calculate cluster centers
        unique_clusters = set(clusters.values())
        num_clusters = len(unique_clusters)
        
        # Arrange cluster centers in a circle
        for i, cluster_id in enumerate(unique_clusters):
            angle = 2 * np.pi * i / num_clusters
            radius = self.config.cluster_spacing * 2
            cluster_centers[cluster_id] = (
                radius * np.cos(angle),
                radius * np.sin(angle)
            )
        
        # Position nodes within their clusters
        for cluster_id in unique_clusters:
            cluster_nodes = [node for node, cid in clusters.items() if cid == cluster_id]
            center_x, center_y = cluster_centers[cluster_id]
            
            if len(cluster_nodes) == 1:
                positions[cluster_nodes[0]] = (center_x, center_y)
            else:
                # Arrange nodes in a small circle around cluster center
                for j, node in enumerate(cluster_nodes):
                    sub_angle = 2 * np.pi * j / len(cluster_nodes)
                    sub_radius = self.config.node_spacing
                    
                    x = center_x + sub_radius * np.cos(sub_angle)
                    y = center_y + sub_radius * np.sin(sub_angle)
                    positions[node] = (x, y)
        
        self.node_positions = positions
        return positions
    
    def _adjust_positions_for_clusters(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Adjust positions to better separate clusters"""
        if not self.node_clusters:
            return positions
        
        adjusted_positions = positions.copy()
        
        # Calculate cluster centroids
        cluster_centroids = {}
        cluster_nodes = defaultdict(list)
        
        for node, cluster_id in self.node_clusters.items():
            cluster_nodes[cluster_id].append(node)
        
        for cluster_id, nodes in cluster_nodes.items():
            if nodes:
                centroid_x = np.mean([positions[node][0] for node in nodes])
                centroid_y = np.mean([positions[node][1] for node in nodes])
                cluster_centroids[cluster_id] = (centroid_x, centroid_y)
        
        # Separate cluster centroids
        unique_clusters = list(cluster_centroids.keys())
        if len(unique_clusters) > 1:
            for i, cluster_id in enumerate(unique_clusters):
                angle = 2 * np.pi * i / len(unique_clusters)
                target_x = self.config.cluster_spacing * 3 * np.cos(angle)
                target_y = self.config.cluster_spacing * 3 * np.sin(angle)
                
                current_x, current_y = cluster_centroids[cluster_id]
                offset_x = target_x - current_x
                offset_y = target_y - current_y
                
                # Apply offset to all nodes in cluster
                for node in cluster_nodes[cluster_id]:
                    old_x, old_y = adjusted_positions[node]
                    adjusted_positions[node] = (old_x + offset_x, old_y + offset_y)
        
        return adjusted_positions
    
    def _apply_collision_detection(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply collision detection to prevent node overlapping"""
        adjusted_positions = positions.copy()
        nodes = list(positions.keys())
        
        # Iterative collision resolution
        for iteration in range(10):  # Max iterations to prevent infinite loops
            collision_detected = False
            
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    pos1 = np.array(adjusted_positions[node1])
                    pos2 = np.array(adjusted_positions[node2])
                    
                    distance = np.linalg.norm(pos1 - pos2)
                    min_distance = self.config.collision_margin
                    
                    if distance < min_distance and distance > 0:
                        collision_detected = True
                        
                        # Calculate separation vector
                        direction = (pos1 - pos2) / distance
                        separation_needed = min_distance - distance
                        
                        # Move nodes apart
                        move_distance = separation_needed / 2
                        adjusted_positions[node1] = tuple(pos1 + direction * move_distance)
                        adjusted_positions[node2] = tuple(pos2 - direction * move_distance)
            
            if not collision_detected:
                break
        
        return adjusted_positions
    
    def visualize(self, save_path: Optional[str] = None, show_plot: bool = True) -> plt.Figure:
        """Create enhanced visualization of the graph"""
        if not self.graph:
            raise ValueError("No graph loaded. Call load_data() first.")
        
        # Apply clustering and layout
        clusters = self.apply_clustering()
        positions = self.apply_layout()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw edges first (so they appear behind nodes)
        edge_list = list(self.graph.edges())
        for edge in edge_list:
            node1, node2 = edge
            pos1 = positions[node1]
            pos2 = positions[node2]
            
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   'k-', alpha=self.config.edge_alpha, 
                   linewidth=self.config.edge_width_base)
        
        # Draw nodes grouped by cluster/type
        unique_clusters = set(clusters.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_nodes = [node for node, cid in clusters.items() if cid == cluster_id]
            
            if not cluster_nodes:
                continue
            
            # Get representative node type for cluster
            node_types_in_cluster = [self.node_types.get(node, 'Unknown') for node in cluster_nodes]
            most_common_type = max(set(node_types_in_cluster), key=node_types_in_cluster.count)
            
            # Get color for this node type
            color = self.config.node_type_colors.get(most_common_type, 
                                                   self.config.node_type_colors['default'])
            
            # Plot nodes in this cluster
            x_coords = [positions[node][0] for node in cluster_nodes]
            y_coords = [positions[node][1] for node in cluster_nodes]
            
            ax.scatter(x_coords, y_coords, 
                      c=color, s=self.config.node_size_base,
                      alpha=0.8, edgecolors='black', linewidth=1,
                      label=f'Cluster {cluster_id} ({most_common_type})')
        
        # Add labels for important nodes (optional, to avoid clutter)
        # for node, pos in positions.items():
        #     ax.annotate(node[:8], pos, xytext=(5, 5), textcoords='offset points',
        #                fontsize=8, alpha=0.7)
        
        ax.set_title(f'Enhanced Graph Visualization\n'
                    f'Layout: {self.config.layout_type.value}, '
                    f'Clustering: {self.config.clustering_strategy.value}', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def compare_layouts(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comparison of different layout algorithms"""
        if not self.graph:
            raise ValueError("No graph loaded. Call load_data() first.")
        
        layout_types = [LayoutType.FORCE_DIRECTED, LayoutType.HIERARCHICAL, 
                       LayoutType.CIRCULAR, LayoutType.GRID]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, layout_type in enumerate(layout_types):
            # Temporarily change layout type
            original_layout = self.config.layout_type
            self.config.layout_type = layout_type
            
            # Apply clustering and layout
            clusters = self.apply_clustering()
            positions = self.apply_layout()
            
            ax = axes[i]
            
            # Draw edges
            for edge in self.graph.edges():
                node1, node2 = edge
                pos1 = positions[node1]
                pos2 = positions[node2]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'k-', alpha=0.3, linewidth=0.5)
            
            # Draw nodes
            unique_clusters = set(clusters.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for j, cluster_id in enumerate(unique_clusters):
                cluster_nodes = [node for node, cid in clusters.items() if cid == cluster_id]
                
                if cluster_nodes:
                    x_coords = [positions[node][0] for node in cluster_nodes]
                    y_coords = [positions[node][1] for node in cluster_nodes]
                    
                    ax.scatter(x_coords, y_coords, c=[colors[j]], s=50,
                              alpha=0.8, edgecolors='black', linewidth=0.5)
            
            ax.set_title(f'{layout_type.value.replace("_", " ").title()} Layout', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Restore original layout
            self.config.layout_type = original_layout
        
        plt.suptitle('Layout Algorithm Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layout comparison saved to {save_path}")
        
        plt.show()
        return fig


def main():
    """Main function to demonstrate the enhanced graph plotting system"""
    
    # Create configuration
    config = LayoutConfig(
        layout_type=LayoutType.FORCE_DIRECTED,
        clustering_strategy=ClusteringStrategy.NODE_TYPE,
        node_spacing=80.0,
        cluster_spacing=300.0,
        enable_collision_detection=True
    )
    
    # Create plotter
    plotter = EnhancedGraphPlotter(config)
    
    # Load data
    if not plotter.load_data('0708烟台S4_updated.json'):
        print("Failed to load data")
        return
    
    # Create enhanced visualization
    print("\nCreating enhanced visualization...")
    plotter.visualize(save_path='enhanced_graph_plot.png')
    
    # Create layout comparison
    print("\nCreating layout comparison...")
    plotter.compare_layouts(save_path='layout_comparison.png')
    
    # Test different clustering strategies
    print("\nTesting different clustering strategies...")
    strategies = [ClusteringStrategy.NODE_TYPE, ClusteringStrategy.SPATIAL, 
                 ClusteringStrategy.HYBRID]
    
    for strategy in strategies:
        config.clustering_strategy = strategy
        plotter.config = config
        plotter.visualize(save_path=f'clustering_{strategy.value}.png', show_plot=False)
        print(f"  Created visualization with {strategy.value} clustering")


if __name__ == "__main__":
    main()