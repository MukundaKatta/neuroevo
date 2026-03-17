"""NetworkGenome - encodes neural architectures as genes."""

from __future__ import annotations

import random
from typing import Optional

from neuroevo.models import Gene, Genome, LayerType


class InnovationTracker:
    """Global tracker for structural innovation numbers (NEAT-style)."""

    def __init__(self) -> None:
        self._counter = 0
        self._history: dict[tuple[int, int], int] = {}

    def get_innovation(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._history:
            self._counter += 1
            self._history[key] = self._counter
        return self._history[key]

    @property
    def current(self) -> int:
        return self._counter


_global_tracker = InnovationTracker()


def get_tracker() -> InnovationTracker:
    return _global_tracker


def reset_tracker() -> None:
    global _global_tracker
    _global_tracker = InnovationTracker()


class NetworkGenome:
    """Encodes a neural network architecture as a collection of genes."""

    _next_id = 0

    def __init__(self, input_size: int = 10, output_size: int = 2) -> None:
        NetworkGenome._next_id += 1
        self.genome_id = NetworkGenome._next_id
        self.input_size = input_size
        self.output_size = output_size
        self.node_genes: list[Gene] = []
        self.connection_genes: list[Gene] = []
        self._next_node = 0
        self._init_minimal()

    def _init_minimal(self) -> None:
        """Create minimal topology: input -> output."""
        tracker = get_tracker()
        in_node = self._add_node(LayerType.LINEAR, self.input_size, self.input_size)
        out_node = self._add_node(LayerType.LINEAR, self.input_size, self.output_size)
        innov = tracker.get_innovation(in_node, out_node)
        conn = Gene(
            gene_id=len(self.connection_genes),
            gene_type="connection",
            src_node=in_node,
            dst_node=out_node,
            weight=random.gauss(0, 1),
            enabled=True,
            innovation=innov,
        )
        self.connection_genes.append(conn)

    def _add_node(
        self,
        layer_type: LayerType,
        in_size: int,
        out_size: int,
    ) -> int:
        node_id = self._next_node
        self._next_node += 1
        gene = Gene(
            gene_id=node_id,
            gene_type="node",
            layer_type=layer_type,
            in_size=in_size,
            out_size=out_size,
        )
        self.node_genes.append(gene)
        return node_id

    def add_hidden_node(
        self,
        layer_type: LayerType = LayerType.LINEAR,
        size: Optional[int] = None,
    ) -> int:
        """Mutate by splitting a connection and inserting a hidden node."""
        if size is None:
            size = random.choice([8, 16, 32, 64])
        enabled_conns = [c for c in self.connection_genes if c.enabled]
        if not enabled_conns:
            return -1
        conn = random.choice(enabled_conns)
        conn.enabled = False
        tracker = get_tracker()
        src = conn.src_node
        dst = conn.dst_node
        new_node = self._add_node(layer_type, size, size)
        innov1 = tracker.get_innovation(src, new_node)
        innov2 = tracker.get_innovation(new_node, dst)
        self.connection_genes.append(
            Gene(
                gene_id=len(self.connection_genes),
                gene_type="connection",
                src_node=src,
                dst_node=new_node,
                weight=1.0,
                enabled=True,
                innovation=innov1,
            )
        )
        self.connection_genes.append(
            Gene(
                gene_id=len(self.connection_genes),
                gene_type="connection",
                src_node=new_node,
                dst_node=dst,
                weight=conn.weight,
                enabled=True,
                innovation=innov2,
            )
        )
        return new_node

    def add_connection(self) -> bool:
        """Mutate by adding a new connection between existing nodes."""
        if len(self.node_genes) < 2:
            return False
        tracker = get_tracker()
        src = random.choice(self.node_genes)
        dst = random.choice(self.node_genes)
        if src.gene_id == dst.gene_id:
            return False
        existing = {(c.src_node, c.dst_node) for c in self.connection_genes}
        if (src.gene_id, dst.gene_id) in existing:
            return False
        innov = tracker.get_innovation(src.gene_id, dst.gene_id)
        self.connection_genes.append(
            Gene(
                gene_id=len(self.connection_genes),
                gene_type="connection",
                src_node=src.gene_id,
                dst_node=dst.gene_id,
                weight=random.gauss(0, 1),
                enabled=True,
                innovation=innov,
            )
        )
        return True

    def mutate_weights(self, perturb_rate: float = 0.8) -> None:
        """Mutate connection weights."""
        for conn in self.connection_genes:
            if random.random() < perturb_rate:
                conn.weight += random.gauss(0, 0.3)
            else:
                conn.weight = random.gauss(0, 1)

    def to_genome(self) -> Genome:
        return Genome(
            genome_id=self.genome_id,
            node_genes=list(self.node_genes),
            connection_genes=list(self.connection_genes),
        )

    def copy(self) -> "NetworkGenome":
        new = NetworkGenome.__new__(NetworkGenome)
        NetworkGenome._next_id += 1
        new.genome_id = NetworkGenome._next_id
        new.input_size = self.input_size
        new.output_size = self.output_size
        new._next_node = self._next_node
        new.node_genes = [g.model_copy() for g in self.node_genes]
        new.connection_genes = [g.model_copy() for g in self.connection_genes]
        return new

    @staticmethod
    def compatibility_distance(
        g1: "NetworkGenome",
        g2: "NetworkGenome",
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
    ) -> float:
        """NEAT compatibility distance between two genomes."""
        innovs1 = {c.innovation: c for c in g1.connection_genes}
        innovs2 = {c.innovation: c for c in g2.connection_genes}
        all_innovs = set(innovs1.keys()) | set(innovs2.keys())
        if not all_innovs:
            return 0.0
        max1 = max(innovs1.keys()) if innovs1 else 0
        max2 = max(innovs2.keys()) if innovs2 else 0
        threshold = min(max1, max2)
        disjoint = 0
        excess = 0
        matching_weights: list[float] = []
        for innov in all_innovs:
            in1 = innov in innovs1
            in2 = innov in innovs2
            if in1 and in2:
                matching_weights.append(
                    abs(innovs1[innov].weight - innovs2[innov].weight)
                )
            elif innov <= threshold:
                disjoint += 1
            else:
                excess += 1
        n = max(len(g1.connection_genes), len(g2.connection_genes), 1)
        avg_w = sum(matching_weights) / max(len(matching_weights), 1)
        return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * avg_w)

    @staticmethod
    def crossover(
        parent1: "NetworkGenome",
        parent2: "NetworkGenome",
        fitness1: float,
        fitness2: float,
    ) -> "NetworkGenome":
        """NEAT-style crossover: fitter parent's disjoint/excess genes kept."""
        child = parent1.copy() if fitness1 >= fitness2 else parent2.copy()
        better = parent1 if fitness1 >= fitness2 else parent2
        worse = parent1 if fitness1 < fitness2 else parent2
        worse_innovs = {c.innovation: c for c in worse.connection_genes}
        for i, conn in enumerate(child.connection_genes):
            if conn.innovation in worse_innovs:
                if random.random() < 0.5:
                    child.connection_genes[i] = worse_innovs[
                        conn.innovation
                    ].model_copy()
        return child
