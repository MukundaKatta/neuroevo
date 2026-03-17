"""NEAT algorithm with speciation and innovation tracking."""

from __future__ import annotations

import random
from typing import Optional

from neuroevo.evolution.fitness import FitnessEvaluator
from neuroevo.evolution.genome import NetworkGenome, reset_tracker
from neuroevo.evolution.population import Population
from neuroevo.models import Generation, Individual, SearchResult


class Species:
    """A NEAT species grouping similar genomes."""

    _next_id = 0

    def __init__(self, representative: Individual) -> None:
        Species._next_id += 1
        self.species_id = Species._next_id
        self.representative = representative
        self.members: list[Individual] = [representative]
        self.best_fitness = representative.fitness
        self.stagnation = 0

    def add(self, individual: Individual) -> None:
        self.members.append(individual)
        individual.species_id = self.species_id

    def update(self) -> None:
        best = max(self.members, key=lambda m: m.fitness)
        if best.fitness > self.best_fitness:
            self.best_fitness = best.fitness
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.representative = random.choice(self.members)


class NEATAlgorithm:
    """NeuroEvolution of Augmenting Topologies."""

    def __init__(
        self,
        population_size: int = 50,
        input_size: int = 10,
        output_size: int = 2,
        compatibility_threshold: float = 3.0,
        max_stagnation: int = 15,
    ) -> None:
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.compat_threshold = compatibility_threshold
        self.max_stagnation = max_stagnation
        self.species: list[Species] = []
        self.population: Optional[Population] = None

    def _speciate(self) -> None:
        """Assign individuals to species based on compatibility distance."""
        assert self.population is not None
        for sp in self.species:
            sp.members = []
        for ind in self.population.members:
            placed = False
            g_ind = NetworkGenome(self.input_size, self.output_size)
            g_ind.node_genes = list(ind.genome.node_genes)
            g_ind.connection_genes = list(ind.genome.connection_genes)
            for sp in self.species:
                g_rep = NetworkGenome(self.input_size, self.output_size)
                g_rep.node_genes = list(sp.representative.genome.node_genes)
                g_rep.connection_genes = list(
                    sp.representative.genome.connection_genes
                )
                dist = NetworkGenome.compatibility_distance(g_ind, g_rep)
                if dist < self.compat_threshold:
                    sp.add(ind)
                    placed = True
                    break
            if not placed:
                new_sp = Species(ind)
                self.species.append(new_sp)
        self.species = [sp for sp in self.species if sp.members]
        for sp in self.species:
            sp.update()
        self.species = [
            sp for sp in self.species if sp.stagnation < self.max_stagnation
        ]
        if not self.species and self.population.members:
            self.species = [Species(self.population.members[0])]

    def run(
        self,
        evaluator: FitnessEvaluator,
        generations: int = 20,
    ) -> SearchResult:
        """Run NEAT evolution."""
        reset_tracker()
        self.population = Population(
            self.population_size, self.input_size, self.output_size
        )
        gen_records: list[Generation] = []
        total_evals = 0
        for gen in range(generations):
            for ind in self.population.members:
                ind.fitness = evaluator.evaluate(ind.genome)
                total_evals += 1
            self._speciate()
            best = self.population.best
            gen_records.append(
                Generation(
                    gen_number=gen,
                    best_fitness=best.fitness,
                    avg_fitness=self.population.avg_fitness,
                    num_species=len(self.species),
                    best_genome_id=best.genome.genome_id,
                    population_size=len(self.population.members),
                )
            )
            self.population.evolve()
        return SearchResult(
            best_genome=self.population.best.genome,
            best_fitness=self.population.best.fitness,
            generations=gen_records,
            total_evaluations=total_evals,
            algorithm="NEAT",
        )
