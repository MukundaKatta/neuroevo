"""Report generation for evolution runs."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from neuroevo.models import SearchResult


def print_report(result: SearchResult, console: Console | None = None) -> None:
    """Print a rich-formatted report of evolution results."""
    if console is None:
        console = Console()
    console.rule(f"[bold blue]{result.algorithm} Search Report")
    console.print(f"Total evaluations: [cyan]{result.total_evaluations}[/]")
    console.print(f"Best fitness: [green]{result.best_fitness:.4f}[/]")
    console.print(
        f"Best genome nodes: [yellow]{len(result.best_genome.node_genes)}[/]"
    )
    console.print(
        f"Best genome connections: [yellow]{len(result.best_genome.connection_genes)}[/]"
    )
    if result.generations:
        table = Table(title="Generation History")
        table.add_column("Gen", justify="right")
        table.add_column("Best Fitness", justify="right")
        table.add_column("Avg Fitness", justify="right")
        table.add_column("Species", justify="right")
        step = max(1, len(result.generations) // 10)
        for gen in result.generations[::step]:
            table.add_row(
                str(gen.gen_number),
                f"{gen.best_fitness:.4f}",
                f"{gen.avg_fitness:.4f}",
                str(gen.num_species),
            )
        last = result.generations[-1]
        table.add_row(
            str(last.gen_number),
            f"{last.best_fitness:.4f}",
            f"{last.avg_fitness:.4f}",
            str(last.num_species),
        )
        console.print(table)
