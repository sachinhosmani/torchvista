from collections import defaultdict
import torch.nn as nn

from .enums import NodeType


def transform_to_nested_graph(adj_list, node_to_ancestors):
    """
    Transforms a flat adjacency list into a nested graph structure based on node ancestry,
    while preserving all original node and edge information.

    Args:
        adj_list: Dict with structure {node_name: {'edges': [...], 'failed': bool, 'node_type': str}}
        node_to_ancestors: Dict with structure {node_name: [ancestor1, ancestor2, ...]}
                        where ancestors are ordered from immediate parent to root

    Returns:
        Nested dict where each node can contain 'subgraphs' (child modules)
    """

    # Utility function to get the element in the list before the target, if one is present
    def get_element_before(lst, target):
        try:
            index = lst.index(target)
            return lst[index - 1] if index - 1 >= 0 else None
        except ValueError:
            return None

    # Finds the lowest common ancestor between 2 paths, assuming that the paths
    # are ordered from bottom to top (immediate parent first)
    def find_lca(path1, path2):
        lca = None
        for a, b in zip(path1[::-1], path2[::-1]):
            if a == b:
                lca = a
            else:
                break
        return lca

    # Given two nodes, determines the correct pair of "representative" nodes
    # and returns them for linking in the nested graph
    def get_representative_nodes(node1, node2):
        ancestry1, ancestry2 = node_to_ancestors.get(node1, []), node_to_ancestors.get(node2, [])

        # Special cases when the LCA cannot be found
        if not ancestry1 and not ancestry2:
            return node1, node2
        elif not ancestry1:
            return node1, ancestry2[-1]
        elif not ancestry2:
            return ancestry1[-1], node2
        else:
            # When LCA is likely to be present
            lca = find_lca(ancestry1, ancestry2)
            if not lca:
                # This can happen if the 2 nodes have completely disjoint hierarchy paths
                return ancestry1[-1], ancestry2[-1]

            # The node just below the LCA in each node serves as the "representative" node
            representative_node1 = get_element_before(ancestry1, lca)
            representative_node2 = get_element_before(ancestry2, lca)

            # If the two nodes are in the same subtree at the same level, they
            # will act as their own representative nodes
            representative_node1 = node1 if representative_node1 is None else representative_node1
            representative_node2 = node2 if representative_node2 is None else representative_node2

            return representative_node1, representative_node2

    # Step 1: Create the basic structure for all nodes (including ancestors)
    # Collect all unique nodes (actual nodes + all ancestors)
    all_nodes = set(adj_list.keys())
    for ancestors in node_to_ancestors.values():
        all_nodes.update(ancestors)

    # Pre-compute original incoming/outgoing dims for each node from the flat adj_list
    # This must be done BEFORE edge redirection so we capture the true edge dimensions
    original_incoming_dims = defaultdict(list)
    original_outgoing_dims = defaultdict(list)
    for source_node, node_data in adj_list.items():
        for edge in node_data.get('edges', []):
            target_node = edge['target']
            dims = edge.get('dims', '')
            original_outgoing_dims[source_node].append(dims)
            original_incoming_dims[target_node].append(dims)

    # For container nodes (ancestors), compute their original dims by looking at
    # edges that cross their boundary. A container's incoming dims are the dims of
    # edges entering any of its descendant nodes from outside the container.
    # Similarly for outgoing dims.
    def get_descendants(container_node):
        """Get all leaf nodes that have this container in their ancestry."""
        descendants = set()
        for node, ancestors in node_to_ancestors.items():
            if container_node in ancestors:
                descendants.add(node)
        return descendants

    ancestor_nodes = all_nodes - set(adj_list.keys())
    for container in ancestor_nodes:
        descendants = get_descendants(container)
        # Incoming: edges from non-descendants to descendants
        # Deduplicate by edge_data_id so same tensor fanning out to multiple internal nodes counts once
        seen_incoming = set()
        for target_node in descendants:
            for source_node, node_data in adj_list.items():
                if source_node in descendants:
                    continue
                for edge in node_data.get('edges', []):
                    if edge['target'] == target_node:
                        edge_data_id = edge.get('edge_data_id')
                        if edge_data_id is not None:
                            if edge_data_id in seen_incoming:
                                continue
                            seen_incoming.add(edge_data_id)
                        original_incoming_dims[container].append(edge.get('dims', ''))
        # Outgoing: edges from descendants to non-descendants
        # Deduplicate by edge_data_id so same tensor going to multiple external targets counts once
        seen_outgoing = set()
        for source_node in descendants:
            for edge in adj_list.get(source_node, {}).get('edges', []):
                target_node = edge['target']
                if target_node not in descendants:
                    edge_data_id = edge.get('edge_data_id')
                    if edge_data_id is not None:
                        if edge_data_id in seen_outgoing:
                            continue
                        seen_outgoing.add(edge_data_id)
                    original_outgoing_dims[container].append(edge.get('dims', ''))

    # Initialize nested structure for each node
    nodes = {}
    for node in all_nodes:
        if node in adj_list:
            # Copy all original data from adj_list
            nodes[node] = {
                'edges': [],
                'subgraphs': {},
                'failed': adj_list[node].get('failed', False),
                'node_type': adj_list[node].get('node_type', NodeType.MODULE.value),
                'original_incoming_dims': tuple(sorted(original_incoming_dims.get(node, []))),
                'original_outgoing_dims': tuple(sorted(original_outgoing_dims.get(node, []))),
            }
        else:
            # For ancestor nodes not in adj_list (container modules)
            nodes[node] = {
                'edges': [],
                'subgraphs': {},
                'failed': False,
                'node_type': NodeType.MODULE.value,
                'original_incoming_dims': tuple(sorted(original_incoming_dims.get(node, []))),
                'original_outgoing_dims': tuple(sorted(original_outgoing_dims.get(node, []))),
            }

    # Step 2: Process edges and redirect them to representative nodes
    # Track seen edges to avoid duplicates when the same tensor flows to multiple
    # consumers inside a nested module (e.g., residual connections where x goes to
    # both a layer and an add node). Key: (rep_source, rep_target, edge_data_id)
    seen_edges = set()

    for source_node, node_data in adj_list.items():
        for edge in node_data['edges']:
            target_node = edge['target']

            # Get representative nodes for this edge
            rep_source, rep_target = get_representative_nodes(source_node, target_node)

            # Skip duplicate edges from the same tensor between the same representative nodes
            edge_data_id = edge.get('edge_data_id')
            if edge_data_id is not None:
                edge_key = (rep_source, rep_target, edge_data_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

            dims = edge.get('dims', '')

            # Create new edge dict with all original information
            new_edge = {
                'target': rep_target,
                'dims': dims,
            }

            # Preserve optional edge attributes (only if they exist and are not None)
            if edge_data_id is not None:
                new_edge['edge_data_id'] = edge_data_id
            if 'is_implied_edge' in edge:
                new_edge['is_implied_edge'] = edge['is_implied_edge']

            # Add edge to representative source node
            nodes[rep_source]['edges'].append(new_edge)

    # Step 3: Build the nested hierarchy
    # Keep track of the nodes that stay at root level
    root = dict(nodes)

    # Nest each node under its immediate parent
    for node, ancestors in node_to_ancestors.items():
        if ancestors:
            # Nest this node and all its ancestors appropriately
            for child, parent in zip([node] + list(ancestors[:-1]), ancestors):
                if child in nodes and parent in nodes:
                    # Move child into parent's subgraphs
                    nodes[parent]['subgraphs'][child] = nodes[child]
                    # Remove from root level
                    root.pop(child, None)

    return root


def transform_to_unnested_graph(nested_graph, node_to_ancestors):
    """
    Reverses ``transform_to_nested_graph`` by flattening a nested graph back into an
    adjacency list whose nodes are the deepest leaves. Edges between containers are
    redirected to the ingress/egress leaves of those containers.

    Args:
        nested_graph: Nested dict structure produced by ``transform_to_nested_graph``.
        node_to_ancestors: Existing ancestor map for leaf nodes (used for context).

    Returns:
        A tuple of (unnested_adj_list, rebuilt_ancestors) where rebuilt_ancestors
        includes synthesized containers (e.g. repeat_*).
    """
    # Collect references to every node in the nested graph for quick lookup and rebuild
    # the ancestor map (immediate parent first) including any synthesized containers.
    all_nodes = {}
    rebuilt_ancestors = defaultdict(list)

    def collect_nodes(subgraph, ancestors):
        for node_name, node_data in subgraph.items():
            all_nodes[node_name] = node_data
            rebuilt_ancestors[node_name] = list(reversed(ancestors))
            collect_nodes(node_data.get('subgraphs', {}), ancestors + [node_name])

    collect_nodes(nested_graph, [])

    # Initialize the flat adjacency list for every leaf node
    unnested_adj_list = {}
    for node_name, node_data in all_nodes.items():
        if not node_data.get('subgraphs'):
            unnested_adj_list[node_name] = {
                'edges': [],
                'failed': node_data.get('failed', False),
                'node_type': node_data.get('node_type', NodeType.MODULE.value),
            }

    # Memoized ingress/egress lookups for each node
    ingress_cache = {}
    egress_cache = {}

    def _internal_edge_counts(subgraph):
        incoming = defaultdict(int)
        outgoing = defaultdict(int)
        for child_name, child_data in subgraph.items():
            for edge in child_data.get('edges', []):
                target = edge['target']
                if target in subgraph:
                    outgoing[child_name] += 1
                    incoming[target] += 1
        for child in subgraph:
            incoming.setdefault(child, 0)
            outgoing.setdefault(child, 0)
        return incoming, outgoing

    def get_ingress_leaves(node_name):
        if node_name in ingress_cache:
            return ingress_cache[node_name]

        node_data = all_nodes.get(node_name, {})
        subgraph = node_data.get('subgraphs', {})

        if not subgraph:
            ingress_cache[node_name] = [node_name]
            return ingress_cache[node_name]

        incoming, _ = _internal_edge_counts(subgraph)
        ingress_candidates = [n for n, count in incoming.items() if count == 0]
        if not ingress_candidates:
            ingress_candidates = list(subgraph.keys())

        leaves = []
        for child in ingress_candidates:
            leaves.extend(get_ingress_leaves(child))

        ingress_cache[node_name] = leaves
        return leaves

    def get_egress_leaves(node_name):
        if node_name in egress_cache:
            return egress_cache[node_name]

        node_data = all_nodes.get(node_name, {})
        subgraph = node_data.get('subgraphs', {})

        if not subgraph:
            egress_cache[node_name] = [node_name]
            return egress_cache[node_name]

        _, outgoing = _internal_edge_counts(subgraph)
        egress_candidates = [n for n, count in outgoing.items() if count == 0]
        if not egress_candidates:
            egress_candidates = list(subgraph.keys())

        leaves = []
        for child in egress_candidates:
            leaves.extend(get_egress_leaves(child))

        egress_cache[node_name] = leaves
        return leaves

    # Redirect every edge to operate on the deepest nodes
    for source_name, source_data in all_nodes.items():
        source_leaves = get_egress_leaves(source_name)
        if not source_leaves:
            continue

        for edge in source_data.get('edges', []):
            target_name = edge['target']
            target_leaves = get_ingress_leaves(target_name) if target_name in all_nodes else []
            if not target_leaves:
                continue

            for src_leaf in source_leaves:
                for tgt_leaf in target_leaves:
                    new_edge = {
                        'target': tgt_leaf,
                        'dims': edge.get('dims', ''),
                    }
                    if 'edge_data_id' in edge:
                        new_edge['edge_data_id'] = edge['edge_data_id']
                    if 'is_implied_edge' in edge:
                        new_edge['is_implied_edge'] = edge['is_implied_edge']

                    # Only populate edges for true leaves
                    if src_leaf in unnested_adj_list:
                        unnested_adj_list[src_leaf]['edges'].append(new_edge)

    return unnested_adj_list, rebuilt_ancestors


def inject_modulelist_containers(nested_graph, module_hierarchy, module_to_node_name, graph_node_name_to_without_suffix, graph_node_display_names, node_to_module_path):
    """
    Validates and injects ModuleList containers into the nested graph where children form valid chains.
    Processes the nested graph recursively and injects ModuleLists at the appropriate levels.
    """
    import re
    modulelist_counter = 0

    def extract_trailing_number(name):
        """Extract trailing number from node name for numeric sorting."""
        match = re.search(r'_(\d+)$', name)
        return int(match.group(1)) if match else 0

    def get_modulelist_parent_path(modulelist_instance):
        """Compute the path to a ModuleList's parent in the nested graph."""
        path = []
        current = module_hierarchy.get(modulelist_instance)

        while current is not None:
            if current in module_to_node_name:
                # Use the last node name if module was called multiple times
                node_names = module_to_node_name[current]
                path.insert(0, node_names[-1] if isinstance(node_names, list) else node_names)
            current = module_hierarchy.get(current)

        return path

    processed_modulelists = set()

    def validate_and_inject_at_level(graph_dict, current_path, modulelists_at_path):
        """
        Recursively process graph levels and inject ModuleLists where valid.
        Returns: modified graph_dict
        """
        nonlocal modulelist_counter

        # Check if any ModuleList should exist at this level
        for ml_instance, ml_path in modulelists_at_path.items():
            if ml_path != current_path:
                continue
            if id(ml_instance) in processed_modulelists:
                continue
            processed_modulelists.add(id(ml_instance))

            # Find children of this ModuleList
            children_modules = [m for m, p in module_hierarchy.items() if p == ml_instance]
            # Get all node names for each child module (flatten list of lists)
            children_names = []
            for m in children_modules:
                if m in module_to_node_name:
                    node_names = module_to_node_name[m]
                    if isinstance(node_names, list):
                        children_names.extend(node_names)
                    else:
                        children_names.append(node_names)

            if not children_names:
                continue

            # Check all children exist at this level
            all_present = all(child in graph_dict for child in children_names)
            if not all_present:
                continue

            ordered_children = sorted(children_names, key=extract_trailing_number)

            # Validate chain: each child should connect to the next
            is_valid_chain = True
            for i in range(len(ordered_children) - 1):
                current_node = ordered_children[i]
                next_node = ordered_children[i + 1]

                edges = graph_dict[current_node].get('edges', [])
                targets = [e['target'] for e in edges]

                if (len(targets) > 1) or (next_node not in targets):
                    is_valid_chain = False
                    break

            if not is_valid_chain:
                continue

            # Valid chain! Create ModuleList container
            ml_name = f"ModuleList_{modulelist_counter}"
            modulelist_counter += 1

            # Add metadata
            graph_node_name_to_without_suffix[ml_name] = "ModuleList"
            graph_node_display_names[ml_name] = "ModuleList"
            node_to_module_path[ml_name] = "torch.nn.modules.container"

            # Create container with children as subgraphs
            ml_container = {
                'edges': [],
                'subgraphs': {},
                'failed': False,
                'node_type': NodeType.MODULE.value,
            }

            # Move children into container
            for child_name in ordered_children:
                ml_container['subgraphs'][child_name] = graph_dict[child_name]

            # Find edges from last child to nodes outside the ModuleList
            last_child = ordered_children[-1]
            for edge in graph_dict[last_child].get('edges', []):
                if edge['target'] not in ordered_children:
                    # Create clean edge copy (exclude None values to avoid JS null errors)
                    new_edge = {
                        'target': edge['target'],
                        'dims': edge.get('dims', ''),
                    }
                    if edge.get('edge_data_id') is not None:
                        new_edge['edge_data_id'] = edge['edge_data_id']
                    if 'is_implied_edge' in edge:
                        new_edge['is_implied_edge'] = edge['is_implied_edge']
                    ml_container['edges'].append(new_edge)

            # Remove children from current level
            for child_name in ordered_children:
                del graph_dict[child_name]

            # Add container to current level
            graph_dict[ml_name] = ml_container

        # Recursively process subgraphs
        for node_name in list(graph_dict.keys()):
            node_data = graph_dict[node_name]
            if 'subgraphs' in node_data and node_data['subgraphs']:
                # Skip injected ModuleLists when building path (they weren't in the original hierarchy)
                if graph_node_name_to_without_suffix.get(node_name) == 'ModuleList':
                    new_path = current_path
                else:
                    new_path = current_path + [node_name]
                validate_and_inject_at_level(node_data['subgraphs'], new_path, modulelists_at_path)

    # Collect all ModuleLists with their parent paths
    modulelists_with_paths = {}
    for module_instance in module_hierarchy.keys():
        if isinstance(module_instance, nn.ModuleList):
            parent_path = get_modulelist_parent_path(module_instance)
            modulelists_with_paths[module_instance] = parent_path

    # Process the nested graph
    validate_and_inject_at_level(nested_graph, [], modulelists_with_paths)


def compress_nested_graph(nested_graph, adj_list, node_to_module_path, graph_node_name_to_without_suffix, graph_node_display_names):
    """
    Compresses a nested graph by finding repeating patterns in Sequential/ModuleList containers
    and replacing them with repeat_<count>_<counter> containers.

    Args:
        nested_graph: Nested dict structure from transform_to_nested_graph
        module_info: Dict containing module information
        func_info: Dict containing function information
        node_to_module_path: Dict mapping node names to module paths
        graph_node_name_to_without_suffix: Dict mapping node names to display names

    Returns:
        Compressed nested graph in the exact same format as input
    """

    repeat_counter = 0  # Global counter for unique repeat container names
    repeat_containers = set()
    signature_cache = {}  # Memoization cache for structural signatures

    def get_chain_from_subgraph(subgraph):
        """
        Extracts the linear chain of nodes from a subgraph (for Sequential/ModuleList).
        Returns list of node names in order.
        The subgraph IS already an adjacency list!
        """
        nodes = list(subgraph.keys())
        if not nodes:
            return []

        # Find source node (node with no incoming edges)
        incoming_count = {node: 0 for node in nodes}
        for node in nodes:
            for edge in subgraph[node]['edges']:
                target = edge['target']
                if target in incoming_count:
                    incoming_count[target] += 1

        source_nodes = [node for node, count in incoming_count.items() if count == 0]
        if not source_nodes:
            # Fallback: just use first node
            return [nodes[0]]

        # Trace the chain from source
        chain = []
        current = source_nodes[0]
        visited = set()

        while current and current not in visited:
            chain.append(current)
            visited.add(current)

            # Find next node in chain by looking at edges
            edges = subgraph[current]['edges']
            next_node = None
            for edge in edges:
                target = edge['target']
                # Only follow edge if target is in this subgraph
                if target in subgraph and target not in visited:
                    next_node = target
                    break
            current = next_node

        return chain

    def serialize_subgraph(subgraph):
        """
        Convert a subgraph dict into a canonical hashable tuple that captures
        the full topology: node types, module paths, edges with dims, and
        nested subgraphs recursively.

        Sorts node names by extracting trailing number (e.g., Linear_5 -> 5)
        so structurally identical subgraphs serialize to the same tuple.
        """
        if not subgraph:
            return ()

        import re
        def extract_number(name):
            # Handle scalars_{id}_x_{count} pattern - extract the id, not the count
            scalars_match = re.match(r'scalars_(\d+)_x_\d+$', name)
            if scalars_match:
                return int(scalars_match.group(1))
            # Default: extract trailing number
            match = re.search(r'_(\d+)$', name)
            return int(match.group(1)) if match else 0

        # Sort by global ID for canonical ordering
        sorted_names = sorted(subgraph.keys(), key=extract_number)
        name_to_index = {name: i for i, name in enumerate(sorted_names)}

        serialized_nodes = []
        for name in sorted_names:
            node_data = subgraph[name]
            node_type = graph_node_name_to_without_suffix.get(name, name)
            module_path = node_to_module_path.get(name, '')

            # Serialize edges as (target_index, dims) - use -1 for external targets
            edges = tuple(sorted(
                (name_to_index.get(e['target'], -1), e.get('dims', ''))
                for e in node_data.get('edges', [])
            ))

            # Recursively serialize nested subgraphs
            child_serial = serialize_subgraph(node_data.get('subgraphs', {}))

            serialized_nodes.append((node_type, module_path, edges, child_serial))

        return tuple(serialized_nodes)

    def get_structural_signature(node_name, node_data, parent_subgraph):
        """
        Creates a structural signature for a node that captures its type,
        module path, incoming/outgoing dims, and full subgraph topology.
        Two nodes with identical signatures are structurally equivalent.

        Uses the original_incoming_dims and original_outgoing_dims stored on each node
        (computed before edge redirection in transform_to_nested_graph) to ensure
        boundary nodes in a container are compared correctly with interior nodes.
        """
        if node_name in signature_cache:
            return signature_cache[node_name]

        node_type = graph_node_name_to_without_suffix.get(node_name, node_name)
        module_path = node_to_module_path.get(node_name, '')

        # Use the original dims stored on the node (computed before nesting redirected edges)
        incoming_dims = node_data.get('original_incoming_dims', ())
        outgoing_dims = node_data.get('original_outgoing_dims', ())

        # Serialize the full subgraph topology
        subgraph_serial = serialize_subgraph(node_data.get('subgraphs', {}))

        signature = (node_type, module_path, incoming_dims, outgoing_dims, subgraph_serial)
        signature_cache[node_name] = signature
        return signature

    def nodes_are_equivalent(node1, node2, parent_subgraph):
        """
        Checks if two nodes are structurally equivalent by comparing their signatures.
        """
        # Quick check: same type and module path first
        type1 = graph_node_name_to_without_suffix.get(node1, node1)
        type2 = graph_node_name_to_without_suffix.get(node2, node2)
        if type1 != type2:
            return False

        path1 = node_to_module_path.get(node1, '')
        path2 = node_to_module_path.get(node2, '')
        if path1 != path2:
            return False

        # Full structural comparison
        node1_data = parent_subgraph.get(node1, {})
        node2_data = parent_subgraph.get(node2, {})

        sig1 = get_structural_signature(node1, node1_data, parent_subgraph)
        sig2 = get_structural_signature(node2, node2_data, parent_subgraph)

        return sig1 == sig2

    def compress_chain(chain, parent_subgraph, adj_list):
        """
        Compresses a chain by finding repeating consecutive nodes.
        Returns new subgraph structure with compressed nodes.
        Each node in the result is also recursively compressed.
        """
        nonlocal repeat_counter

        if len(chain) <= 1:
            # Still need to recursively process the single node
            if len(chain) == 1:
                node = chain[0]
                return {node: process_node(node, parent_subgraph[node])}
            return {}

        new_subgraph = {}
        i = 0

        while i < len(chain):
            current_node = chain[i]
            repeat_count = 1

            # Count how many times this node repeats consecutively
            j = i + 1
            while j < len(chain) and nodes_are_equivalent(current_node, chain[j], parent_subgraph):
                repeat_count += 1
                j += 1

            if repeat_count > 1:
                # Create a repeat container
                repeat_name = f"repeat_{repeat_count}_{repeat_counter}"
                repeat_counter += 1
                repeat_containers.add(repeat_name)

                # Get the first node's data and RECURSIVELY PROCESS IT
                first_node = chain[i]
                first_node_data = parent_subgraph[first_node]
                processed_first_node = process_node(first_node, first_node_data)
                processed_first_node['edges'] = []  # Edges will be handled by the repeat container

                # Create the repeat container
                new_subgraph[repeat_name] = {
                    'edges': [],
                    'subgraphs': {
                        first_node: processed_first_node
                    },
                    'failed': False,
                    'node_type': NodeType.MODULE.value,
                }

                # Update metadata for repeat container
                graph_node_name_to_without_suffix[repeat_name] = f"REPEAT X {repeat_count}"
                graph_node_display_names[repeat_name] = f"REPEAT {repeat_count}x"
                node_to_module_path[repeat_name] = ""

                # Add edge from repeat container to next node (if exists)
                if j < len(chain):
                    next_node = chain[j]
                    # Get dims from the last repeated node's edge to next
                    last_repeated_node = chain[j - 1]
                    last_node_edges = parent_subgraph[last_repeated_node]['edges']

                    # Find edge that goes to next_node
                    dims = '( )'
                    edge_data_id = None
                    is_implied = False
                    for edge in last_node_edges:
                        dims = edge['dims']
                        edge_data_id = edge.get('edge_data_id')
                        is_implied = edge.get('is_implied_edge', False)
                        break

                    new_edge = {
                        'target': next_node,
                        'dims': dims,
                    }
                    if edge_data_id is not None:
                        new_edge['edge_data_id'] = edge_data_id
                    if is_implied:
                        new_edge['is_implied_edge'] = is_implied

                    new_subgraph[repeat_name]['edges'].append(new_edge)

                # Redirect the previous node's edge to point to repeat container instead of first_node
                if i > 0:
                    prev_node = chain[i - 1]
                    # prev_node might itself be inside a repeat container we just created
                    # Find which key in new_subgraph contains prev_node
                    prev_key = None
                    for key in new_subgraph.keys():
                        if key == prev_node:
                            prev_key = key
                            break
                        elif key.startswith('repeat_') and prev_node in new_subgraph[key]['subgraphs']:
                            prev_key = key
                            break

                    if prev_key:
                        # Update its edge that points to first_node to point to repeat_name instead
                        for edge in new_subgraph[prev_key]['edges']:
                            if edge['target'] == first_node:
                                edge['target'] = repeat_name

                i = j
            else:
                # Not repeating, RECURSIVELY PROCESS the node and copy it
                node_data = parent_subgraph[current_node]
                new_subgraph[current_node] = process_node(current_node, node_data)
                i += 1


        return new_subgraph

    def process_node(node_name, node_data):
        """
        Recursively processes a node and its subgraphs.
        Returns the processed node data.
        """
        # Check if this is a Sequential or ModuleList
        node_display_name = graph_node_name_to_without_suffix.get(node_name, node_name)
        is_sequential_or_modulelist = node_display_name in ['Sequential', 'ModuleList']

        new_node_data = {
            'edges': node_data['edges'].copy(),
            'subgraphs': {},
            'failed': node_data.get('failed', False),
            'node_type': node_data.get('node_type', NodeType.MODULE.value),
        }

        if is_sequential_or_modulelist and node_data['subgraphs']:
            # Extract chain and compress it
            # compress_chain will recursively process each node in the chain
            chain = get_chain_from_subgraph(node_data['subgraphs'])
            new_node_data['subgraphs'] = compress_chain(chain, node_data['subgraphs'], adj_list)
        else:
            # Not Sequential/ModuleList, just recursively process subgraphs
            for sub_node_name, sub_node_data in node_data['subgraphs'].items():
                new_node_data['subgraphs'][sub_node_name] = process_node(sub_node_name, sub_node_data)

        return new_node_data

    # Process the root level
    compressed_graph = {}
    for node_name, node_data in nested_graph.items():
        compressed_graph[node_name] = process_node(node_name, node_data)

    return compressed_graph, repeat_containers


def build_immediate_ancestor_map(ancestor_dict, adj_list):
    immediate_ancestor_map = {}
    for node, ancestors in ancestor_dict.items():
        if ancestors and node in adj_list:
            immediate_ancestor_map[node] = ancestors[0]
            for i in range(len(ancestors) - 1):
                if ancestors[i] not in immediate_ancestor_map:
                    immediate_ancestor_map[ancestors[i]] = ancestors[i + 1]
    return immediate_ancestor_map

