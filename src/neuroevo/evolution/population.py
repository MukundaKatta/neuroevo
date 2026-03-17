"""Population management with selection, crossover, and mutation."""

from __future__ import annotations

import random

from neuroevo.evolution.genome import NetworkGenome
from neuroevo.models import Individual


class Population:
    """Manages a population of neural architecture genomes."""

    def __init__(
        self,
        size: int = 50,
        input_size: int = 10,
        output_size: int = 2,
    ) -> None:
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.members: list[Individual] = []
        self.generation = 0
        self._init_population()

    def _init_population(self) -> None:
        for i in range(self.size):
            genome = NetworkGenome(self.input_size, self.output_size)
            self.members.append(
                Individual(
                    individual_id=i,
                    genome=genome.to_genome(),
                    generation=0,
                )
            )

    def tournament_select(self, k: int = 3) -> Individual:
        """Tournament selection: pick k random, return the fittest."""
        contestants = random.sample(self.members, min(k, len(self.members)))
        return max(contestants, key=lambda ind: ind.fitness)

    def roulette_select(self) -> Individual:
        """Fitness-proportionate selection."""
        fitnesses = [max(ind.fitness, 0.0) for ind in self.members]
        total = sum(fitnesses)
        if total == 0:
            return random.choice(self.members)
        pick = random.uniform(0, total)
        current = 0.0
        for ind, f in zip(self.members, fitnesses):
            current += f
            if current >= pick:
                return ind
        return self.members[-1]

    def crossover(self, parent1: Individual, parent2: Individual) -> NetworkGenome:
        """Perform crossover between two parents."""
        g1 = NetworkGenome(self.input_size, self.output_size)
        g1.node_genes = [g.model_copy() for g in parent1.genome.node_genes]
        g1.connection_genes = [g.model_copy() for g in parent1.genome.connection_genes]
        g1._next_node = len(g1.node_genes)

        g2 = NetworkGenome(self.input_size, self.output_size)
        g2.node_genes = [g.model_copy() for g in parent2.genome.node_genes]
        g2.connection_genes = [g.model_copy() for g in parent2.genome.connection_genes]
        g2._next_node = len(g2.node_genes)

        return NetworkGenome.crossover(g1, g2, parent1.fitness, parent2.fitness)

    def mutate(
        self,
        genome: NetworkGenome,
        add_node_rate: float = 0.03,
        add_conn_rate: float = 0.05,
        weight_rate: float = 0.8,
    ) -> NetworkGenome:
        """Apply mutation operators."""
        if random.random() < add_node_rate:
            genome.add_hidden_node()
        if random.random() < add_conn_rate:
            genome.add_connection()
        if random.random() < weight_rate:
            genome.mutate_weights()
        return genome

    def evolve(self, elitism: int = 2) -> None:
        """Create next generation via selection, crossover, mutation."""
        self.generation += 1
        self.members.sort(key=lambda ind: ind.fitness, reverse=True)
        new_members: list[Individual] = []
        # Elitism
        for i, elite in enumerate(self.members[:elitism]):
            new_members.append(
                Individual(
                    individual_id=i,
                    genome=elite.genome.model_copy(deep=True),
                    fitness=elite.fitness,
                    generation=self.generation,
                    species_id=elite.species_id,
                )
            )
        # Fill rest with offspring
        while len(new_members) < self.size:
            p1 = self.tournament_select()
            p2 = self.tournament_select()
            child_genome = self.crossover(p1, p2)
            child_genome = self.mutate(child_genome)
            idx = len(new_members)
            new_members.append(
                Individual(
                    individual_id=idx,
                    genome=child_genome.to_genome(),
                    generation=self.generation,
                )
            )
        self.members = new_members

    @property
    def best(self) -> Individual:
        return max(self.members, key=lambda ind: ind.fitness)

    @property
    def avg_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(ind.fitness for ind in self.members) / len(self.members)
