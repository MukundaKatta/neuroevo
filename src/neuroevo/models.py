"""Pydantic models for neural architecture evolution."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LayerType(str, Enum):
    LINEAR = "linear"
    CONV2D = "conv2d"
    BATCHNORM = "batchnorm"
    RELU = "relu"
    DROPOUT = "dropout"
    MAXPOOL = "maxpool"
    FLATTEN = "flatten"


class Gene(BaseModel):
    """A single gene encoding a network component."""

    gene_id: int
    gene_type: str = Field(description="'node' or 'connection'")
    layer_type: Optional[LayerType] = None
    in_size: Optional[int] = None
    out_size: Optional[int] = None
    src_node: Optional[int] = None
    dst_node: Optional[int] = None
    weight: float = 1.0
    enabled: bool = True
    innovation: int = 0


class Genome(BaseModel):
    """Complete genome encoding a neural architecture."""

    genome_id: int
    node_genes: list[Gene] = Field(default_factory=list)
    connection_genes: list[Gene] = Field(default_factory=list)
    fitness: float = 0.0
    species_id: int = 0
    age: int = 0


class Individual(BaseModel):
    """An individual in the population."""

    individual_id: int
    genome: Genome
    fitness: float = 0.0
    generation: int = 0
    species_id: int = 0


class Generation(BaseModel):
    """Record of a single generation."""

    gen_number: int
    best_fitness: float
    avg_fitness: float
    num_species: int
    best_genome_id: int
    population_size: int


class SearchResult(BaseModel):
    """Result of an architecture search."""

    best_genome: Genome
    best_fitness: float
    generations: list[Generation] = Field(default_factory=list)
    total_evaluations: int = 0
    algorithm: str = ""
