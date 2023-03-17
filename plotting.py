import networkx as nx
import matplotlib.pyplot as plt


def plot_graph( graph, communities = None, pos = None, edge_color = "0.5", edge_width=0.5, node_size=100 ):
	if pos is None:
		if any( "pos" not in graph.nodes[u] for u in graph ):
			pos = nx.spring_layout( graph )
		else:
			pos = { u: graph.nodes[u]["pos"] for u in graph}
	
	if communities is None:
		node_color = [ 1 for i in range( nx.number_of_nodes( graph ) ) ]
	else:
		node_color = communities
	
	nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        cmap=plt.get_cmap("tab20"),
    )
	nx.draw_networkx_edges( graph, pos = pos, width = edge_width, edge_color = edge_color )




def plot_single_partition(
    graph, all_results, scale_id, edge_color="0.5", edge_width=0.5, node_size=100
):
    """Plot the community structures for a given scale.
    Args:
        graph (networkx.Graph): graph to plot
        all_results (dict): results of pygenstability scan
        scale_id (int): index of scale to plot
        folder (str): folder to save figures
        edge_color (str): color of edges
        edge_width (float): width of edges
        node_size (float): size of nodes
        ext (str): extension of figures files
    """
    if any("pos" not in graph.nodes[u] for u in graph):
        pos = nx.spring_layout(graph)
        for u in graph:
            graph.nodes[u]["pos"] = pos[u]

    pos = {u: graph.nodes[u]["pos"] for u in graph}

    node_color = all_results["community_id"][scale_id]

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        cmap=plt.get_cmap("tab20"),
    )
    nx.draw_networkx_edges(graph, pos=pos, width=edge_width, edge_color=edge_color)

    plt.axis("off")
    plt.title(
        str(r"$log_{10}(scale) =$ ")
        + str(np.round(np.log10(all_results["scales"][scale_id]), 2))
        + ", with "
        + str(all_results["number_of_communities"][scale_id])
        + " communities"
    )
