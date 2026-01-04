from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import heapq


class RelationType(str, Enum):
    CAUSES = "causes"
    TREATS = "treats"
    SYMPTOM_OF = "symptom_of"
    PREVENTS = "prevents"
    RELATED_TO = "related_to"
    SUBSIDIZES = "subsidizes"
    REQUIRES = "requires"
    APPLIES_TO = "applies_to"


@dataclass
class Edge:
    target: str
    relation: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    id: str
    node_type: str
    name: str
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._adjacency: Dict[str, List[Edge]] = {}
        self._reverse_adjacency: Dict[str, List[Tuple[str, Edge]]] = {}
        self._type_index: Dict[str, Set[str]] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if node_id in self._nodes:
            return

        node = Node(id=node_id, node_type=node_type, name=name, metadata=metadata or {})
        self._nodes[node_id] = node
        self._adjacency[node_id] = []
        self._reverse_adjacency[node_id] = []

        if node_type not in self._type_index:
            self._type_index[node_type] = set()
        self._type_index[node_type].add(node_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        weight: float = 1.0,
        metadata: Dict[str, Any] | None = None,
        bidirectional: bool = False,
    ) -> None:
        if source_id not in self._nodes or target_id not in self._nodes:
            return

        edge = Edge(
            target=target_id, relation=relation, weight=weight, metadata=metadata or {}
        )

        self._adjacency[source_id].append(edge)
        self._reverse_adjacency[target_id].append((source_id, edge))

        if bidirectional:
            reverse_edge = Edge(
                target=source_id,
                relation=relation,
                weight=weight,
                metadata=metadata or {},
            )
            self._adjacency[target_id].append(reverse_edge)
            self._reverse_adjacency[source_id].append((target_id, reverse_edge))

    def get_neighbors(self, node_id: str) -> List[Tuple[str, Edge]]:
        if node_id not in self._adjacency:
            return []
        return [(edge.target, edge) for edge in self._adjacency[node_id]]

    def get_incoming(self, node_id: str) -> List[Tuple[str, Edge]]:
        return self._reverse_adjacency.get(node_id, [])

    def bfs(
        self,
        start_id: str,
        max_depth: int = 3,
        node_filter: Callable[[Node], bool] | None = None,
        relation_filter: Set[RelationType] | None = None,
    ) -> List[Tuple[str, int, List[str]]]:
        if start_id not in self._nodes:
            return []

        visited: Set[str] = {start_id}
        queue: deque = deque([(start_id, 0, [start_id])])
        results: List[Tuple[str, int, List[str]]] = []

        while queue:
            current_id, depth, path = queue.popleft()

            if depth > 0:
                node = self._nodes[current_id]
                if node_filter is None or node_filter(node):
                    results.append((current_id, depth, path))

            if depth >= max_depth:
                continue

            for neighbor_id, edge in self.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue

                if relation_filter and edge.relation not in relation_filter:
                    continue

                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1, path + [neighbor_id]))

        return results

    def dfs(
        self,
        start_id: str,
        max_depth: int = 5,
        node_filter: Callable[[Node], bool] | None = None,
        relation_filter: Set[RelationType] | None = None,
    ) -> List[Tuple[str, int, List[str]]]:
        """
        Depth-First Search from start node.
        Returns: List of (node_id, depth, path) tuples.
        Explores deeper before wider - good for finding complete paths.
        Time: O(V + E)
        """
        if start_id not in self._nodes:
            return []

        results: List[Tuple[str, int, List[str]]] = []
        visited: Set[str] = set()

        def _dfs_recursive(current_id: str, depth: int, path: List[str]) -> None:
            if current_id in visited or depth > max_depth:
                return

            visited.add(current_id)

            if depth > 0:
                node = self._nodes[current_id]
                if node_filter is None or node_filter(node):
                    results.append((current_id, depth, path.copy()))

            for neighbor_id, edge in self.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue
                if relation_filter and edge.relation not in relation_filter:
                    continue
                _dfs_recursive(neighbor_id, depth + 1, path + [neighbor_id])

        _dfs_recursive(start_id, 0, [start_id])
        return results

    def dijkstra(
        self, start_id: str, end_id: str | None = None, max_cost: float = float("inf")
    ) -> Dict[str, Tuple[float, List[str]]]:
        if start_id not in self._nodes:
            return {}

        distances: Dict[str, float] = {start_id: 0}
        paths: Dict[str, List[str]] = {start_id: [start_id]}
        pq: List[Tuple[float, str]] = [(0, start_id)]
        visited: Set[str] = set()

        while pq:
            current_dist, current_id = heapq.heappop(pq)

            if current_id in visited:
                continue

            visited.add(current_id)
            if end_id and current_id == end_id:
                break

            if current_dist > max_cost:
                continue

            for neighbor_id, edge in self.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue

                new_dist = current_dist + edge.weight

                if new_dist < distances.get(neighbor_id, float("inf")):
                    distances[neighbor_id] = new_dist
                    paths[neighbor_id] = paths[current_id] + [neighbor_id]
                    heapq.heappush(pq, (new_dist, neighbor_id))

        return {node_id: (distances[node_id], paths[node_id]) for node_id in distances}

    def find_path(
        self, start_id: str, end_id: str, use_weights: bool = False
    ) -> Optional[List[str]]:
        if use_weights:
            result = self.dijkstra(start_id, end_id)
            return result.get(end_id, (None, None))[1]
        results = self.bfs(start_id, max_depth=10)
        for node_id, _, path in results:
            if node_id == end_id:
                return path
        return None

    def multi_hop_query(
        self, start_id: str, target_types: List[str], max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        results = []

        def type_filter(node: Node) -> bool:
            return node.node_type in target_types

        bfs_results = self.bfs(start_id, max_depth=max_hops, node_filter=type_filter)

        for node_id, depth, path in bfs_results:
            node = self._nodes[node_id]
            results.append(
                {
                    "id": node_id,
                    "name": node.name,
                    "type": node.node_type,
                    "hops": depth,
                    "path": [self._nodes[p].name for p in path],
                    "metadata": node.metadata,
                }
            )

        return results

    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        node_ids = self._type_index.get(node_type, set())
        return [self._nodes[nid] for nid in node_ids]

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_stats(self) -> Dict[str, Any]:
        total_edges = sum(len(edges) for edges in self._adjacency.values())
        return {
            "total_nodes": len(self._nodes),
            "total_edges": total_edges,
            "node_types": {ntype: len(ids) for ntype, ids in self._type_index.items()},
            "avg_degree": total_edges / max(len(self._nodes), 1),
        }


class AgriculturalKnowledgeGraph(KnowledgeGraph):
    def __init__(self):
        super().__init__()
        self._build_agricultural_graph()

    def _build_agricultural_graph(self) -> None:
        diseases = [
            (
                "citrus_canker",
                "Citrus Canker",
                {"severity": "high", "type": "bacterial"},
            ),
            (
                "hlb",
                "Huanglongbing (HLB)",
                {"severity": "critical", "type": "bacterial"},
            ),
            ("tristeza", "Citrus Tristeza", {"severity": "high", "type": "viral"}),
            ("scab", "Citrus Scab", {"severity": "medium", "type": "fungal"}),
            ("melanose", "Melanose", {"severity": "medium", "type": "fungal"}),
            ("gummosis", "Gummosis", {"severity": "high", "type": "fungal"}),
            ("root_rot", "Root Rot", {"severity": "high", "type": "fungal"}),
            ("anthracnose", "Anthracnose", {"severity": "medium", "type": "fungal"}),
        ]

        for disease_id, name, meta in diseases:
            self.add_node(disease_id, "disease", name, meta)
        symptoms = [
            ("yellowing", "Leaf Yellowing"),
            ("lesions", "Raised Lesions"),
            ("fruit_drop", "Premature Fruit Drop"),
            ("dieback", "Branch Dieback"),
            ("chlorosis", "Chlorosis"),
            ("necrosis", "Necrosis"),
            ("wilting", "Wilting"),
            ("gumming", "Gum Exudation"),
            ("mottling", "Leaf Mottling"),
            ("stunting", "Tree Stunting"),
        ]

        for symptom_id, name in symptoms:
            self.add_node(symptom_id, "symptom", name)
        treatments = [
            ("copper_spray", "Copper Fungicide Spray", {"chemical": True}),
            ("bordeaux_mixture", "Bordeaux Mixture", {"chemical": True}),
            ("remove_infected", "Remove Infected Parts", {"chemical": False}),
            ("neem_oil", "Neem Oil Application", {"chemical": False}),
            ("imidacloprid", "Imidacloprid Treatment", {"chemical": True}),
            ("carbendazim", "Carbendazim Application", {"chemical": True}),
            ("ipm", "Integrated Pest Management", {"chemical": False}),
            ("biological_control", "Biological Control Agents", {"chemical": False}),
        ]

        for treatment_id, name, meta in treatments:
            self.add_node(treatment_id, "treatment", name, meta)
        pests = [
            ("asian_citrus_psyllid", "Asian Citrus Psyllid"),
            ("citrus_leafminer", "Citrus Leaf Miner"),
            ("aphids", "Aphids"),
            ("mites", "Spider Mites"),
            ("whitefly", "Whitefly"),
            ("scale_insects", "Scale Insects"),
        ]

        for pest_id, name in pests:
            self.add_node(pest_id, "pest", name)
        schemes = [
            ("pmksy", "PM Krishi Sinchai Yojana", {"type": "irrigation"}),
            ("nhm", "National Horticulture Mission", {"type": "horticulture"}),
            ("pmfby", "PM Fasal Bima Yojana", {"type": "insurance"}),
            ("pm_kisan", "PM-KISAN", {"type": "income_support"}),
            ("kcc", "Kisan Credit Card", {"type": "credit"}),
            ("rkvy", "Rashtriya Krishi Vikas Yojana", {"type": "development"}),
        ]

        for scheme_id, name, meta in schemes:
            self.add_node(scheme_id, "scheme", name, meta)
        disease_symptoms = {
            "citrus_canker": ["lesions", "fruit_drop", "dieback"],
            "hlb": ["yellowing", "mottling", "stunting", "fruit_drop"],
            "tristeza": ["yellowing", "dieback", "stunting"],
            "scab": ["lesions", "fruit_drop"],
            "melanose": ["lesions"],
            "gummosis": ["gumming", "dieback"],
            "root_rot": ["wilting", "yellowing", "dieback"],
            "anthracnose": ["lesions", "fruit_drop", "necrosis"],
        }

        for disease_id, symptom_list in disease_symptoms.items():
            for symptom_id in symptom_list:
                self.add_edge(disease_id, symptom_id, RelationType.CAUSES, weight=0.5)
                self.add_edge(
                    symptom_id, disease_id, RelationType.SYMPTOM_OF, weight=0.5
                )
        disease_treatments = {
            "citrus_canker": ["copper_spray", "bordeaux_mixture", "remove_infected"],
            "hlb": ["imidacloprid", "remove_infected", "biological_control"],
            "tristeza": ["remove_infected", "biological_control"],
            "scab": ["copper_spray", "carbendazim"],
            "melanose": ["copper_spray"],
            "gummosis": ["bordeaux_mixture", "remove_infected"],
            "root_rot": ["carbendazim", "ipm"],
            "anthracnose": ["copper_spray", "carbendazim"],
        }

        for disease_id, treatment_list in disease_treatments.items():
            for treatment_id in treatment_list:
                self.add_edge(treatment_id, disease_id, RelationType.TREATS, weight=0.3)
        pest_diseases = {
            "asian_citrus_psyllid": ["hlb"],
            "citrus_leafminer": ["citrus_canker"],
            "aphids": ["tristeza"],
        }

        for pest_id, disease_list in pest_diseases.items():
            for disease_id in disease_list:
                self.add_edge(pest_id, disease_id, RelationType.CAUSES, weight=0.4)
        scheme_relations = {
            "pmfby": ["citrus_canker", "hlb", "tristeza"],
            "nhm": [
                "copper_spray",
                "ipm",
                "biological_control",
            ],
            "pmksy": ["root_rot", "gummosis"],
        }

        for scheme_id, related in scheme_relations.items():
            for related_id in related:
                if related_id in self._nodes:
                    node_type = self._nodes[related_id].node_type
                    if node_type == "disease":
                        self.add_edge(
                            scheme_id, related_id, RelationType.APPLIES_TO, weight=0.6
                        )
                    elif node_type == "treatment":
                        self.add_edge(
                            scheme_id, related_id, RelationType.SUBSIDIZES, weight=0.4
                        )

    def find_treatments_for_symptom(self, symptom_id: str) -> List[Dict[str, Any]]:
        return self.multi_hop_query(symptom_id, ["treatment"], max_hops=2)

    def find_schemes_for_disease(self, disease_id: str) -> List[Dict[str, Any]]:
        results = []
        for scheme_id in self._type_index.get("scheme", set()):
            path = self.find_path(scheme_id, disease_id)
            if path and len(path) <= 3:
                scheme = self._nodes[scheme_id]
                results.append(
                    {
                        "id": scheme_id,
                        "name": scheme.name,
                        "type": scheme.metadata.get("type"),
                        "path": [self._nodes[p].name for p in path],
                    }
                )

        return results

    def get_disease_context(self, disease_id: str) -> Dict[str, Any]:
        if disease_id not in self._nodes:
            return {}

        disease = self._nodes[disease_id]

        return {
            "disease": disease.name,
            "metadata": disease.metadata,
            "symptoms": self.multi_hop_query(disease_id, ["symptom"], max_hops=1),
            "treatments": self.multi_hop_query(disease_id, ["treatment"], max_hops=2),
            "related_pests": self.multi_hop_query(disease_id, ["pest"], max_hops=2),
            "applicable_schemes": self.find_schemes_for_disease(disease_id),
        }
_agri_kg: AgriculturalKnowledgeGraph | None = None


def get_knowledge_graph() -> AgriculturalKnowledgeGraph:
    global _agri_kg
    if _agri_kg is None:
        _agri_kg = AgriculturalKnowledgeGraph()
    return _agri_kg
