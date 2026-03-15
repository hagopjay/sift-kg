"""Community detection and structural analysis for knowledge graphs.

Provides Louvain-based community detection decoupled from the narrate
pipeline, plus topology analysis (bridges, isolation, inter-community
connections) for agent-consumable structural overviews.
"""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def _build_clean_undirected(kg: KnowledgeGraph) -> nx.Graph:
    """Build undirected graph without DOCUMENT nodes and MENTIONED_IN edges.

    This is the standard graph preparation for community detection and
    topology analysis — strips metadata to expose substantive structure.
    """
    non_doc_nodes = [
        nid for nid, data in kg.graph.nodes(data=True)
        if data.get("entity_type") != "DOCUMENT"
    ]
    subgraph = kg.graph.subgraph(non_doc_nodes).copy()
    edges_to_remove = [
        (u, v, k) for u, v, k, d in subgraph.edges(keys=True, data=True)
        if d.get("relation_type") == "MENTIONED_IN"
    ]
    subgraph.remove_edges_from(edges_to_remove)
    return subgraph.to_undirected()


def detect_communities(
    kg: KnowledgeGraph,
    described_ids: set[str] | None = None,
    min_community_size: int = 8,
) -> list[list[dict[str, Any]]] | None:
    """Detect thematic communities using Louvain method.

    Args:
        kg: Knowledge graph to analyze.
        described_ids: If provided, only include entities with these IDs in
            community membership. If None, all non-DOCUMENT entities are
            included. Used by narrate to filter to described entities.
        min_community_size: Minimum members for a community to be included.

    Returns:
        List of communities (each a list of entity dicts with id, name,
        entity_type) sorted by total degree, or None if detection fails
        or produces <=1 community.
    """
    try:
        undirected = _build_clean_undirected(kg)
        raw_communities = nx.community.louvain_communities(undirected)
    except Exception as e:
        logger.debug(f"Community detection failed: {e}")
        return None

    if len(raw_communities) <= 1:
        return None

    # Build entity dicts for non-DOCUMENT nodes
    entity_map: dict[str, dict[str, Any]] = {}
    for nid, data in kg.graph.nodes(data=True):
        if data.get("entity_type") == "DOCUMENT":
            continue
        entity_map[nid] = {
            "id": nid,
            "name": data.get("name", nid),
            "entity_type": data.get("entity_type", "UNKNOWN"),
        }

    degree_map = dict(kg.graph.degree())

    result: list[list[dict[str, Any]]] = []
    for community_nodes in raw_communities:
        members = []
        for nid in community_nodes:
            if nid not in entity_map:
                continue
            if described_ids is not None and nid not in described_ids:
                continue
            members.append(entity_map[nid])
        if len(members) >= min_community_size:
            result.append(members)

    if not result:
        return None

    # Sort by total degree (most connected first)
    result.sort(
        key=lambda c: sum(degree_map.get(e["id"], 0) for e in c),
        reverse=True,
    )
    return result


def save_communities(
    communities: list[list[dict[str, Any]]],
    output_dir: Path,
    labels: dict[int, str] | None = None,
) -> Path:
    """Save community assignments to communities.json.

    Args:
        communities: List of communities (each a list of entity dicts).
        output_dir: Directory to write communities.json.
        labels: Optional map of community index → label string.
            If None, generates "Community 1", "Community 2", etc.

    Returns:
        Path to written communities.json.
    """
    comm_data: dict[str, str] = {}
    for i, community in enumerate(communities):
        label = (labels or {}).get(i, f"Community {i + 1}")
        for e in community:
            comm_data[e["id"]] = label

    comm_path = output_dir / "communities.json"
    comm_path.write_text(
        json.dumps(comm_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info(f"Communities saved ({len(set(comm_data.values()))} communities)")
    return comm_path


def load_communities(output_dir: Path) -> dict[str, str]:
    """Load community assignments from communities.json.

    Returns:
        Dict mapping entity_id → community label.
        Empty dict if file doesn't exist or is empty.
    """
    comm_path = output_dir / "communities.json"
    if not comm_path.exists():
        return {}
    data = json.loads(comm_path.read_text())
    return data if isinstance(data, dict) else {}


def load_communities_grouped(output_dir: Path) -> dict[str, list[str]]:
    """Load community assignments grouped by label.

    Inverts the entity_id → label mapping to label → list of entity_ids.

    Returns:
        Dict mapping community label → list of entity IDs.
        Empty dict if file doesn't exist.
    """
    flat = load_communities(output_dir)
    grouped: dict[str, list[str]] = {}
    for eid, label in flat.items():
        grouped.setdefault(label, []).append(eid)
    return grouped


def find_bridges(
    kg: KnowledgeGraph,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Find bridge entities that connect 2+ communities.

    Uses undirected projection with DOCUMENT/MENTIONED_IN stripped.

    Args:
        kg: Knowledge graph.
        output_dir: Directory containing communities.json.

    Returns:
        List of bridge dicts with entity, name, entity_type,
        communities, and cross_community_edges.
    """
    community_map = load_communities(output_dir)
    if not community_map:
        return []

    undirected = _build_clean_undirected(kg)
    bridges: list[dict[str, Any]] = []

    for node_id in undirected.nodes():
        if node_id not in community_map:
            continue
        node_community = community_map[node_id]

        neighbor_communities: set[str] = set()
        cross_edges = 0
        for neighbor in undirected.neighbors(node_id):
            neighbor_comm = community_map.get(neighbor)
            if neighbor_comm and neighbor_comm != node_community:
                neighbor_communities.add(neighbor_comm)
                cross_edges += 1

        if neighbor_communities:
            all_communities = sorted({node_community} | neighbor_communities)
            node_data = kg.graph.nodes.get(node_id, {})
            bridges.append({
                "entity": node_id,
                "name": node_data.get("name", node_id),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "communities": all_communities,
                "cross_community_edges": cross_edges,
            })

    bridges.sort(key=lambda b: b["cross_community_edges"], reverse=True)
    return bridges


def find_isolated(kg: KnowledgeGraph) -> list[dict[str, Any]]:
    """Find entities with no substantive connections.

    An entity is isolated if it has degree 0 on the cleaned graph
    (DOCUMENT nodes and MENTIONED_IN edges stripped).

    Args:
        kg: Knowledge graph.

    Returns:
        List of isolated entity dicts with entity, name, entity_type, degree.
    """
    undirected = _build_clean_undirected(kg)
    isolated: list[dict[str, Any]] = []

    for node_id in undirected.nodes():
        if undirected.degree(node_id) == 0:
            node_data = kg.graph.nodes.get(node_id, {})
            isolated.append({
                "entity": node_id,
                "name": node_data.get("name", node_id),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "degree": 0,
            })

    return isolated


def find_community_connections(
    kg: KnowledgeGraph,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Find connections between communities.

    For each pair of communities, counts shared edges and bridge entities.

    Args:
        kg: Knowledge graph.
        output_dir: Directory containing communities.json.

    Returns:
        List of connection dicts with from, to, shared_edges, bridge_entities.
        Sorted by shared_edges descending.
    """
    community_map = load_communities(output_dir)
    if not community_map:
        return []

    undirected = _build_clean_undirected(kg)

    # Count cross-community edges and bridge entities per pair
    pair_edges: dict[tuple[str, str], int] = {}
    pair_bridges: dict[tuple[str, str], set[str]] = {}

    for u, v in undirected.edges():
        u_comm = community_map.get(u)
        v_comm = community_map.get(v)
        if not u_comm or not v_comm or u_comm == v_comm:
            continue
        pair = tuple(sorted([u_comm, v_comm]))
        pair_edges[pair] = pair_edges.get(pair, 0) + 1
        pair_bridges.setdefault(pair, set()).update([u, v])

    connections: list[dict[str, Any]] = []
    for (comm_a, comm_b), edge_count in pair_edges.items():
        bridge_nodes = pair_bridges.get((comm_a, comm_b), set())
        connections.append({
            "from": comm_a,
            "to": comm_b,
            "shared_edges": edge_count,
            "bridge_entities": len(bridge_nodes),
        })

    connections.sort(key=lambda c: c["shared_edges"], reverse=True)
    return connections


def extract_subgraph(
    kg: KnowledgeGraph,
    entity_id: str,
    depth: int = 1,
) -> dict[str, Any]:
    """Extract neighborhood subgraph around an entity.

    Uses BFS on the clean undirected graph (DOCUMENT nodes and MENTIONED_IN
    stripped), then extracts matching nodes and directed edges from the
    original graph.

    Args:
        kg: Knowledge graph.
        entity_id: Center entity ID.
        depth: Number of hops.

    Returns:
        Dict with 'nodes' and 'links' lists matching graph_data.json format.
        Empty lists if entity not found.
    """
    undirected = _build_clean_undirected(kg)
    if entity_id not in undirected:
        return {"nodes": [], "links": []}

    # BFS to collect node set
    visited: set[str] = {entity_id}
    frontier: set[str] = {entity_id}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    # Extract nodes from original graph
    nodes: list[dict[str, Any]] = []
    for nid in visited:
        data = kg.graph.nodes.get(nid, {})
        if data.get("entity_type") == "DOCUMENT":
            continue
        node_dict: dict[str, Any] = {
            "id": nid,
            "entity_type": data.get("entity_type", "UNKNOWN"),
            "name": data.get("name", nid),
            "confidence": data.get("confidence", 1.0),
            "source_documents": data.get("source_documents", []),
        }
        attrs = data.get("attributes")
        if attrs:
            node_dict["attributes"] = attrs
        nodes.append(node_dict)

    # Extract directed edges between visited nodes
    links: list[dict[str, Any]] = []
    for source, target, _key, edata in kg.graph.edges(keys=True, data=True):
        if source not in visited or target not in visited:
            continue
        if edata.get("relation_type") == "MENTIONED_IN":
            continue
        if kg.graph.nodes.get(source, {}).get("entity_type") == "DOCUMENT":
            continue
        if kg.graph.nodes.get(target, {}).get("entity_type") == "DOCUMENT":
            continue
        link_dict: dict[str, Any] = {
            "source": source,
            "target": target,
            "relation_type": edata.get("relation_type", "RELATED_TO"),
            "confidence": edata.get("confidence", 1.0),
        }
        evidence = edata.get("evidence")
        if evidence:
            link_dict["evidence"] = evidence
        support_count = edata.get("support_count")
        if support_count:
            link_dict["support_count"] = support_count
        support_docs = edata.get("support_documents")
        if support_docs:
            link_dict["support_documents"] = support_docs
        links.append(link_dict)

    return {"nodes": nodes, "links": links}


def get_entity_topology(
    kg: KnowledgeGraph,
    entity_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Get topology context for a single entity.

    Looks up community membership and computes bridge status inline
    (without calling find_bridges, which recomputes all bridges).

    Args:
        kg: Knowledge graph.
        entity_id: Entity to look up.
        output_dir: Directory containing communities.json.

    Returns:
        Dict with community, is_bridge, bridge_communities.
    """
    community_map = load_communities(output_dir)

    entity_community = community_map.get(entity_id)
    if not entity_community:
        return {
            "community": None,
            "is_bridge": False,
            "bridge_communities": [],
        }

    # Inline bridge check for this single entity
    undirected = _build_clean_undirected(kg)
    other_communities: set[str] = set()
    if entity_id in undirected:
        for neighbor in undirected.neighbors(entity_id):
            neighbor_comm = community_map.get(neighbor)
            if neighbor_comm and neighbor_comm != entity_community:
                other_communities.add(neighbor_comm)

    return {
        "community": entity_community,
        "is_bridge": len(other_communities) > 0,
        "bridge_communities": sorted(other_communities),
    }
