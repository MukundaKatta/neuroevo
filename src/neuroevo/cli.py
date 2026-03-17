"""CLI for NeuroEvo neural architecture evolution."""

from __future__ import annotations

import click
from rich.console import Console

from neuroevo.evolution.fitness import FitnessEvaluator
from neuroevo.report import print_report
from neuroevo.simulator import SyntheticTask

console = Console()


@click.group()
def cli() -> None:
    """NeuroEvo - Neural Architecture Evolution."""
    pass


@cli.command()
@click.option("--pop-size", default=20, help="Population size")
@click.option("--generations", default=10, help="Number of generations")
@click.option("--input-dim", default=10, help="Input dimensionality")
@click.option("--num-classes", default=2, help="Number of classes")
def neat(pop_size: int, generations: int, input_dim: int, num_classes: int) -> None:
    """Run NEAT architecture search."""
    from neuroevo.search.neat import NEATAlgorithm

    console.print("[bold]Running NEAT architecture search...[/]")
    task = SyntheticTask(input_dim=input_dim, num_classes=num_classes)
    tx, ty, vx, vy = task.generate()
    evaluator = FitnessEvaluator(input_size=input_dim, output_size=num_classes)
    evaluator.set_data(tx, ty, vx, vy)
    algo = NEATAlgorithm(
        population_size=pop_size, input_size=input_dim, output_size=num_classes
    )
    result = algo.run(evaluator, generations=generations)
    print_report(result, console)


@cli.command()
@click.option("--samples", default=50, help="Number of random samples")
@click.option("--input-dim", default=10, help="Input dimensionality")
@click.option("--num-classes", default=2, help="Number of classes")
def random_search(samples: int, input_dim: int, num_classes: int) -> None:
    """Run random search baseline."""
    from neuroevo.search.random_search import RandomSearchBaseline

    console.print("[bold]Running random search baseline...[/]")
    task = SyntheticTask(input_dim=input_dim, num_classes=num_classes)
    tx, ty, vx, vy = task.generate()
    evaluator = FitnessEvaluator(input_size=input_dim, output_size=num_classes)
    evaluator.set_data(tx, ty, vx, vy)
    algo = RandomSearchBaseline(input_size=input_dim, output_size=num_classes)
    result = algo.run(evaluator, num_samples=samples)
    print_report(result, console)


@cli.command()
@click.option("--pop-size", default=20, help="Population size")
@click.option("--cycles", default=50, help="Number of evolution cycles")
@click.option("--input-dim", default=10, help="Input dimensionality")
@click.option("--num-classes", default=2, help="Number of classes")
def aging(pop_size: int, cycles: int, input_dim: int, num_classes: int) -> None:
    """Run aging evolution."""
    from neuroevo.search.aging import AgingEvolution

    console.print("[bold]Running aging evolution...[/]")
    task = SyntheticTask(input_dim=input_dim, num_classes=num_classes)
    tx, ty, vx, vy = task.generate()
    evaluator = FitnessEvaluator(input_size=input_dim, output_size=num_classes)
    evaluator.set_data(tx, ty, vx, vy)
    algo = AgingEvolution(
        population_size=pop_size, input_size=input_dim, output_size=num_classes
    )
    result = algo.run(evaluator, cycles=cycles)
    print_report(result, console)


if __name__ == "__main__":
    cli()
