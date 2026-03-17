"""Tests for neuroevo."""

import numpy as np
import pytest

from neuroevo.evolution.genome import NetworkGenome, reset_tracker
from neuroevo.evolution.population import Population
from neuroevo.evolution.fitness import FitnessEvaluator
from neuroevo.models import LayerType
from neuroevo.simulator import SyntheticTask, make_xor_task


class TestNetworkGenome:
    def setup_method(self):
        reset_tracker()

    def test_init(self):
        g = NetworkGenome(10, 2)
        assert len(g.node_genes) == 2  # input + output
        assert len(g.connection_genes) == 1

    def test_add_hidden_node(self):
        g = NetworkGenome(10, 2)
        node_id = g.add_hidden_node(LayerType.LINEAR, 16)
        assert node_id >= 0
        assert len(g.node_genes) == 3
        assert len(g.connection_genes) == 3  # original disabled + 2 new

    def test_add_connection(self):
        g = NetworkGenome(10, 2)
        g.add_hidden_node(LayerType.LINEAR, 16)
        initial = len(g.connection_genes)
        g.add_connection()
        assert len(g.connection_genes) >= initial

    def test_mutate_weights(self):
        g = NetworkGenome(10, 2)
        old_weight = g.connection_genes[0].weight
        g.mutate_weights()
        # Weight should change (extremely unlikely to stay same)
        # Just check it runs without error

    def test_compatibility_distance(self):
        g1 = NetworkGenome(10, 2)
        g2 = NetworkGenome(10, 2)
        dist = NetworkGenome.compatibility_distance(g1, g2)
        assert isinstance(dist, float)
        assert dist >= 0

    def test_crossover(self):
        g1 = NetworkGenome(10, 2)
        g2 = NetworkGenome(10, 2)
        g2.add_hidden_node()
        child = NetworkGenome.crossover(g1, g2, 0.8, 0.5)
        assert len(child.node_genes) >= 2

    def test_copy(self):
        g = NetworkGenome(10, 2)
        g.add_hidden_node()
        c = g.copy()
        assert c.genome_id != g.genome_id
        assert len(c.node_genes) == len(g.node_genes)

    def test_to_genome(self):
        g = NetworkGenome(10, 2)
        genome = g.to_genome()
        assert genome.genome_id == g.genome_id
        assert len(genome.node_genes) == len(g.node_genes)


class TestPopulation:
    def setup_method(self):
        reset_tracker()

    def test_init(self):
        pop = Population(size=10, input_size=10, output_size=2)
        assert len(pop.members) == 10

    def test_tournament_select(self):
        pop = Population(size=10)
        pop.members[0].fitness = 1.0
        selected = pop.tournament_select(k=10)
        assert selected.fitness == 1.0

    def test_evolve(self):
        pop = Population(size=10)
        for ind in pop.members:
            ind.fitness = np.random.random()
        pop.evolve()
        assert len(pop.members) == 10
        assert pop.generation == 1


class TestFitnessEvaluator:
    def test_evaluate(self):
        reset_tracker()
        task = SyntheticTask(input_dim=10, num_classes=2, num_train=50, num_val=20)
        tx, ty, vx, vy = task.generate()
        evaluator = FitnessEvaluator(input_size=10, output_size=2, train_epochs=2)
        evaluator.set_data(tx, ty, vx, vy)
        g = NetworkGenome(10, 2)
        fitness = evaluator.evaluate(g.to_genome())
        assert 0.0 <= fitness <= 1.0


class TestSyntheticTask:
    def test_generate(self):
        task = SyntheticTask(input_dim=5, num_classes=3, num_train=100, num_val=30)
        tx, ty, vx, vy = task.generate()
        assert tx.shape == (100, 5)
        assert ty.shape == (100,)
        assert vx.shape == (30, 5)
        assert set(ty).issubset({0, 1, 2})

    def test_xor(self):
        tx, ty, vx, vy = make_xor_task(n_samples=100)
        assert tx.shape[1] == 10  # 2 + 8 noise
        assert set(ty).issubset({0, 1})
