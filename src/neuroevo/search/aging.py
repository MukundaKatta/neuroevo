"""Aging Evolution with tournament selection."""

from __future__ import annotations

import random
from collections import deque

from neuroevo.evolution.fitness import FitnessEvaluator
from neuroevo.evolution.genome import NetworkGenome, reset_tracker
from neuroevo.evolution.population import Population
from neuroevo.models import Generation, Individual, LayerType, SearchResult


class AgingEvolution:
    """Regularized evolution with aging: oldest individuals are removed."""

    def __init__(
        self,
        population_size: int = 50,
        input_size: int = 10,
        output_size: int = 2,
        tournament_size: int = 5,
    ) -> None:
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.tournament_size = tournament_size

    def _mutate_genome(self, genome: NetworkGenome) -> NetworkGenome:
        child = genome.copy()
        mutation = random.choice(["add_node", "add_conn", "weights"])
        if mutation == "add_node":
            child.add_hidden_node(
                layer_type=random.choice([LayerType.LINEAR, LayerType.RELU]),
                size=random.choice([8, 16, 32, 64]),
            )
        elif mutation == "add_conn":
            child.add_connection()
        else:
            child.mutate_weights()
        return child

    def run(
        self,
        evaluator: FitnessEvaluator,
        cycles: int = 200,
    ) -> SearchResult:
        reset_tracker()
        population: deque[Individual] = deque()
        best_fitness = -1.0
        best_individual = None
        gen_records: list[Generation] = []
        # Initialize
        for i in range(self.population_size):
            genome = NetworkGenome(self.input_size, self.output_size)
            g = genome.to_genome()
            fitness = evaluator.evaluate(g)
            ind = Individual(
                individual_id=i, genome=g, fitness=fitness, generation=0
            )
            population.append(ind)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = ind
        for cycle in range(cycles):
            # Tournament selection
            sample = random.sample(list(population), min(self.tournament_size, len(population)))
            parent = max(sample, key=lambda x: x.fitness)
            # Create mutated child
            parent_ng = NetworkGenome(self.input_size, self.output_size)
            parent_ng.node_genes = list(parent.genome.node_genes)
            parent_ng.connection_genes = list(parent.genome.connection_genes)
            parent_ng._next_node = len(parent_ng.node_genes)
            child_ng = self._mutate_genome(parent_ng)
            child_g = child_ng.to_genome()
            child_fitness = evaluator.evaluate(child_g)
            child = Individual(
                individual_id=self.population_size + cycle,
                genome=child_g,
                fitness=child_fitness,
                generation=cycle + 1,
            )
            population.append(child)
            # Remove oldest
            if len(population) > self.population_size:
                population.popleft()
            if child_fitness > best_fitness:
                best_fitness = child_fitness
                best_individual = child
            avg = sum(p.fitness for p in population) / len(population)
            gen_records.append(
                Generation(
                    gen_number=cycle,
                    best_fitness=best_fitness,
                    avg_fitness=avg,
                    num_species=1,
                    best_genome_id=best_individual.genome.genome_id if best_individual else 0,
                    population_size=len(population),
                )
            )
        assert best_individual is not None
        return SearchResult(
            best_genome=best_individual.genome,
            best_fitness=best_fitness,
            generations=gen_records,
            total_evaluations=self.population_size + cycles,
            algorithm="AgingEvolution",
        )
