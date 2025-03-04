"""
Visualization functions for cluster_finder package.

This module contains functions for visualizing clusters, structures, and graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import matplotlib.colors as mcolors


def visualize_graph(graph, structure, tm_indices, material_id=None, formula=None):
    """
    Visualize a connectivity graph of transition metal atoms.
    
    Parameters:
        graph (networkx.Graph): Graph to visualize
        structure (Structure): The pymatgen Structure object
        tm_indices (list): Indices of transition metal atoms
        material_id (str, optional): Material ID for the title
        formula (str, optional): Chemical formula for the title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get positions
    pos = {}
    for i, idx in enumerate(tm_indices):
        pos[i] = structure[idx].coords
    
    # Get element colors and labels
    colors = []
    labels = {}
    for i, idx in enumerate(tm_indices):
        element = structure[idx].specie.symbol
        colors.append(mcolors.CSS4_COLORS.get(element.lower(), 'gray'))
        labels[i] = element
    
    # Draw nodes
    for node, (x, y, z) in pos.items():
        ax.scatter(x, y, z, c=colors[node], s=100, edgecolor='k', alpha=0.7)
        ax.text(x, y, z, labels[node], fontsize=12)
    
    # Draw edges
    for u, v in graph.edges():
        x = np.array([pos[u][0], pos[v][0]])
        y = np.array([pos[u][1], pos[v][1]])
        z = np.array([pos[u][2], pos[v][2]])
        ax.plot(x, y, z, c='black', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    title = "Transition Metal Connectivity Graph"
    if material_id and formula:
        title += f" for {formula} ({material_id})"
    ax.set_title(title)
    
    # Set axis limits based on structure bounds with padding
    bounds = structure.lattice.abc
    ax.set_xlim([0, bounds[0]])
    ax.set_ylim([0, bounds[1]])
    ax.set_zlim([0, bounds[2]])
    
    return fig


def visualize_clusters_in_compound(structure, clusters):
    """
    Visualize clusters within a compound structure.
    
    Parameters:
        structure (Structure): The pymatgen Structure object
        clusters (list): List of cluster dictionaries
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not clusters:
        print("No clusters to visualize")
        return None
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all atoms as small gray spheres
    for site in structure:
        ax.scatter(site.coords[0], site.coords[1], site.coords[2], 
                  c='lightgray', s=20, alpha=0.3)
    
    # Plot each cluster with a distinct color
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, cluster_dict in enumerate(clusters):
        cluster = cluster_dict["sites"]
        color = colors[i % len(colors)]
        
        # Draw cluster atoms
        for site in cluster:
            ax.scatter(site.coords[0], site.coords[1], site.coords[2], 
                      c=color, s=100, edgecolor='black', alpha=0.7)
            ax.text(site.coords[0], site.coords[1], site.coords[2], 
                   site.specie.symbol, fontsize=10)
        
        # Draw cluster centroid
        centroid = cluster_dict["centroid"]
        ax.scatter(centroid[0], centroid[1], centroid[2], 
                  c='red', marker='*', s=200, alpha=0.8)
        
        # Draw connections within cluster
        for i, site1 in enumerate(cluster):
            for site2 in cluster[i+1:]:
                if site1.distance(site2) <= cluster_dict["average_distance"] * 1.1:
                    x = np.array([site1.coords[0], site2.coords[0]])
                    y = np.array([site1.coords[1], site2.coords[1]])
                    z = np.array([site1.coords[2], site2.coords[2]])
                    ax.plot(x, y, z, c=color, alpha=0.7, linewidth=2)
    
    # Set labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Clusters in {structure.composition.reduced_formula} - {len(clusters)} clusters')
    
    # Set axis limits based on structure bounds with padding
    bounds = structure.lattice.abc
    ax.set_xlim([0, bounds[0]])
    ax.set_ylim([0, bounds[1]])
    ax.set_zlim([0, bounds[2]])
    
    return fig


def visualize_cluster_lattice(structure, rotation_matrix=None):
    """
    Visualize a structure with highlighted cluster lattice.
    
    Parameters:
        structure (Structure): The pymatgen Structure object
        rotation_matrix (numpy.ndarray, optional): 3x3 rotation matrix
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Apply rotation if provided
    if rotation_matrix is not None:
        # Convert fractional coordinates to cartesian
        cart_coords = np.array([site.coords for site in structure])
        # Apply rotation
        rotated_coords = np.dot(cart_coords, rotation_matrix)
    else:
        rotated_coords = np.array([site.coords for site in structure])
    
    # Plot atoms
    for i, site in enumerate(structure):
        element = site.specie.symbol
        color = mcolors.CSS4_COLORS.get(element.lower(), 'gray')
        ax.scatter(rotated_coords[i, 0], rotated_coords[i, 1], rotated_coords[i, 2], 
                  c=color, s=80, alpha=0.7, edgecolor='black')
        ax.text(rotated_coords[i, 0], rotated_coords[i, 1], rotated_coords[i, 2], 
               element, fontsize=8)
    
    # Plot unit cell
    lattice = structure.lattice
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    # Convert fractional to cartesian
    vertices_cart = np.dot(vertices, lattice.matrix)
    
    # Apply rotation if provided
    if rotation_matrix is not None:
        vertices_cart = np.dot(vertices_cart, rotation_matrix)
    
    # Define edges of the unit cell
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    # Plot edges
    for start, end in edges:
        ax.plot([vertices_cart[start, 0], vertices_cart[end, 0]],
                [vertices_cart[start, 1], vertices_cart[end, 1]],
                [vertices_cart[start, 2], vertices_cart[end, 2]],
                'k-', alpha=0.5)
    
    # Set labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Structure Visualization: {structure.composition.reduced_formula}')
    
    # Auto-scale axes
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    
    return fig 