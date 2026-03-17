"""Random search baseline for architecture search."""

from __future__ import annotations

import random

from neuroevo.evolution.fitness import FitnessEvaluator
from neuroevo.evolution.genome import NetworkGenome, reset_tracker
from neuroevo.models import Generation, LayerType, SearchResult


class RandomSearchBaseline:
    """Generate random architectures and evaluate them."""

    def __init__(
        self,
        input_size: int = 10,
        output_size: int = 2,
        max_hidden_nodes: int = 5,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.max_hidden_nodes = max_hidden_nodes

    def _random_genome(self) -> NetworkGenome:
        genome = NetworkGenome(self.input_size, self.output_size)
        n_hidden = random.randint(0, self.max_hidden_nodes)
        for _ in range(n_hidden):
            genome.add_hidden_node(
                layer_type=random.choice([LayerType.LINEAR, LayerType.RELU]),
                size=random.choice([8, 16, 32, 64]),
            )
        for _ in range(random.randint(0, 3)):
            genome.add_connection()
        genome.mutate_weights()
        return genome

    def run(
        self,
        evaluator: FitnessEvaluator,
        num_samples: int = 100,
    ) -> SearchResult:
        reset_tracker()
        best_genome = None
        best_fitness = -1.0
        gen_records: list[Generation] = []
        for i in range(num_samples):
            genome = self._random_genome()
            g = genome.to_genome()
            fitness = evaluator.evaluate(g)
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = g
            gen_records.append(
                Generation(
                    gen_number=i,
                    best_fitness=best_fitness,
                    avg_fitness=fitness,
                    num_species=1,
                    best_genome_id=g.genome_id,
                    population_size=1,
                )
            )
        assert best_genome is not None
        return SearchResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            generations=gen_records,
            total_evaluations=num_samples,
            algorithm="RandomSearch",
        )
