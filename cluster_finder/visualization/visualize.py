"""
Visualization functions for cluster_finder package.

This module contains functions for visualizing clusters, structures, and graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import DummySpecies
from ase.visualize.plot import plot_atoms
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
import numpy as np
import matplotlib.colors as mcolors


def visualize_graph(graph, structure=None, tm_indices=None, material_id=None, formula=None, use_3d=False):
    """
    Visualize a connectivity graph of transition metal atoms.
    
    Parameters:
        graph (networkx.Graph): Graph to visualize
        structure (Structure): The pymatgen Structure object
        tm_indices (list): Indices of transition metal atoms
        material_id (str, optional): Material ID for the title
        formula (str, optional): Chemical formula for the title
        use_3d (bool, optional): Whether to use 3D visualization or 2D layout
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if len(graph.edges) < 1:
        print("No edges to visualize")
        fig = plt.figure(figsize=(10, 8))
        return fig
    
    # Create edge weights based on distances
    edge_weights = {
        (u, v): 1 / max(structure.sites[tm_indices[u]].distance(structure.sites[tm_indices[v]]), 1e-5)
        for u, v in graph.edges
    }
    
    # Get element colors and labels
    labels = {i: structure.sites[tm_indices[i]].specie.symbol for i in graph.nodes}
    colors = []
    for i in graph.nodes:
        element = structure.sites[tm_indices[i]].specie.symbol
        colors.append(mcolors.CSS4_COLORS.get(element.lower(), 'skyblue'))
    
    # Create distance labels
    edge_labels = {
        (u, v): f"{structure.sites[tm_indices[u]].distance(structure.sites[tm_indices[v]]):.2f} Å"
        for u, v in graph.edges
    }
    
    if use_3d:
        # 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get real atom positions
        pos = {}
        for i in graph.nodes:
            pos[i] = structure.sites[tm_indices[i]].coords
        
        # Draw nodes
        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, c=colors[node], s=150, edgecolor='black', alpha=0.9)
            ax.text(x, y, z, labels[node], fontsize=12, fontweight='bold', ha='center', va='center')
        
        # Draw edges and edge labels
        for u, v in graph.edges():
            x = np.array([pos[u][0], pos[v][0]])
            y = np.array([pos[u][1], pos[v][1]])
            z = np.array([pos[u][2], pos[v][2]])
            ax.plot(x, y, z, c='gray', alpha=0.7, linewidth=2)
            
            # Add distance label at the middle of the edge
            mid_x, mid_y, mid_z = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2
            ax.text(mid_x, mid_y, mid_z, edge_labels[(u, v)], fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Set labels and title
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        
        # Set axis limits based on structure bounds with padding
        bounds = structure.lattice.abc
        ax.set_xlim([0, bounds[0]])
        ax.set_ylim([0, bounds[1]])
        ax.set_zlim([0, bounds[2]])
    else:
        # 2D visualization using Kamada-Kawai layout
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        pos = nx.kamada_kawai_layout(graph, weight=None)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, node_size=800, node_color=colors, 
            edgecolors="black", alpha=0.9, ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=2.0, alpha=0.7, edge_color="gray", ax=ax)
        
        # Add labels to nodes
        for node, (x, y) in pos.items():
            plt.text(
                x, y, labels[node], fontsize=10, fontweight='bold',
                color="black", ha="center", va="center"
            )
        
        # Add edge labels
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=10, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'), ax=ax
        )
        
        ax.axis("off")
    
    # Set title
    title = "Transition Metal Connectivity Graph"
    if material_id and formula:
        title += f" for {formula} ({material_id})"
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    return fig


def visualize_clusters_in_compound(structure, clusters, cluster_index=None, rotation="45x,30y,0z"):
    """
    Visualizes clusters in the given structure.
    
    Parameters:
        structure (Structure): The Pymatgen Structure object.
        clusters (list[dict]): List of clusters. Each cluster is a dictionary containing:
                               - 'sites': List of Pymatgen Site objects in the cluster.
                               - 'size': Size of the cluster (number of sites).
                               - 'average_distance': Average distance between sites in the cluster.
                               - 'centroid': Centroid coordinates of the cluster.
        cluster_index (int, optional): Index of the specific cluster to visualize.
                                      If None, the first cluster is visualized.
        rotation (str, optional): Rotation parameter for ASE's plot_atoms (e.g., "45x,30y,0z").
    
    Returns:
        matplotlib.figure.Figure: The figure containing the visualization of the selected cluster.
                                  Returns None if no clusters to visualize.
    """
    if not clusters:
        return None
        
    # Convert structure to ASE atoms just once
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    
    # Dictionary to keep track of atom types in the legend (for optimization)
    atom_types = {chemical_symbols[atom.number]: jmol_colors[atom.number] * 0.7 for atom in atoms}
    
    # Select which cluster to visualize
    if cluster_index is not None and 0 <= cluster_index < len(clusters):
        # Use the specified cluster
        cluster = clusters[cluster_index]
    elif len(clusters) > 0:
        # Default to the first cluster
        cluster = clusters[0]
    else:
        return None
        
    # Set atom colors - low saturation for all atoms, high for cluster atoms
    atom_colors = np.array([jmol_colors[atom.number] * 0.7 for atom in atoms])
    
    # Highlight cluster atoms
    try:
        cluster_indices = [structure.sites.index(site) for site in cluster["sites"]]
        for idx in cluster_indices:
            if idx < len(atom_colors):  # Safety check
                atom_colors[idx] = [1.0, 0.0, 0.0]  # High-saturation red for cluster atoms
    except ValueError as e:
        # Handle case where site is not found in structure
        pass
        
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot atoms with specified colors and perspective view
    plot_atoms(atoms, ax, radii=0.5, rotation=rotation, colors=atom_colors)
    
    # Add centroid marker if available
    if "centroid" in cluster:
        # Get the plot transformation
        ax_transform = ax.transData
        # Plot the centroid as a star marker
        centroid = cluster["centroid"]
        ax.scatter(centroid[0], centroid[1], transform=ax_transform, 
                  color='gold', marker='*', s=200, zorder=10, 
                  edgecolor='black', label='Centroid')
                  
    # Add a legend for atom types
    legend_handles = []
    for symbol, color in atom_types.items():
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=symbol)
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[1.0, 0.0, 0.0], markersize=10, label='Cluster Atoms')
    )
    if "centroid" in cluster:
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=10, label='Centroid')
        )
    ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1, 1), title='Atom Types')
    
    # Add cluster number to the title
    title = f"Cluster {cluster_index + 1 if cluster_index is not None else 1}/{len(clusters)}"
    
    # Add a title with cluster info
    ax.set_title(f"{title} - Size: {cluster['size']}, Avg Distance: {cluster['average_distance']:.2f} Å")
    
    # Remove x and y axes
    ax.axis('off')
    plt.tight_layout()
    return fig


def visualize_cluster_lattice(conventional_structure, rot="80x,20y,0z"):
    """
    Visualizes the conventional unit cell with cluster lattice sites labeled as C(i).

    Parameters:
        conventional_structure (Structure): The Pymatgen Structure object representing the generated unit cell.
        rot (str): Rotation parameter for ASE's plot_atoms (e.g., "45x,30y,0z").
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create a copy of the structure to avoid modifying the original
    structure_copy = conventional_structure.copy()

    # Replace DummySpecies with a real element (e.g., 'C')
    new_sites = []
    for site in structure_copy.sites:
        if isinstance(site.specie, DummySpecies):
            # Create a new site with 'C' as the species
            new_site = site.__class__(
                lattice=structure_copy.lattice,
                species='C',
                coords=site.frac_coords,
                properties=site.properties
            )
            new_sites.append(new_site)
        else:
            new_sites.append(site)

    # Update the structure's sites with the new list
    structure_copy._sites = new_sites

    # Convert the structure to an ASE Atoms object
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure_copy)

    # Assign a unique "C(i)" label to each unique cluster site
    unique_sites = {}
    atom_colors = np.array([jmol_colors[atom.number] * 0.5 for atom in atoms])  # Low-saturation colors

    for i, site in enumerate(structure_copy.sites):
        coord_tuple = tuple(np.round(site.frac_coords, 3))  # Use fractional coordinates to identify uniqueness
        if coord_tuple not in unique_sites:
            unique_sites[coord_tuple] = f'C({len(unique_sites) + 1})'  # Assign cluster index

    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot using ASE's plot_atoms
    plot_atoms(atoms, ax, radii=0.5, rotation=rot, colors=atom_colors)

    # Remove axes and display plot
    ax.axis("off")
    plt.title("Cluster Lattice in the Conventional Unit Cell")
    plt.tight_layout()
    return fig
