"""Genetic algorithm solver using the DEAP evolutionary computation framework."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from deap import base, creator, tools

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.genetic.fitness import (
    misplaced_facelets_fitness,
    solved_bonus_fitness,
    weighted_layer_fitness,
)
from rubiks_solve.solvers.genetic.operators import (
    cx_single_point,
    mut_random_move,
    sel_tournament_variable,
)

_FITNESS_STRATEGIES = {
    "misplaced_facelets": misplaced_facelets_fitness,
    "weighted_layer": weighted_layer_fitness,
    "solved_bonus": solved_bonus_fitness,
}

# DEAP requires registering FitnessMax / Individual exactly once per process.
# Guard against re-registration across multiple GeneticSolver instantiations.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


@dataclass
class GeneticConfig:
    """Hyper-parameters for :class:`GeneticSolver`.

    Attributes:
        population_size: Number of chromosomes in each generation.
        max_generations: Maximum number of evolutionary generations to run.
        cx_prob: Probability of applying crossover to a pair of offspring.
        mut_prob: Probability of applying mutation to an individual.
        tournament_size: Number of contestants per tournament selection round.
        min_chromosome_length: Minimum number of moves in an initial chromosome.
        max_chromosome_length: Maximum number of moves in an initial chromosome.
            Defaults to 100; callers should set this to ``puzzle.move_limit()``
            for the target puzzle type.
        fitness_strategy: Name of the fitness function to use.  One of
            ``"misplaced_facelets"``, ``"weighted_layer"``, or
            ``"solved_bonus"``.
        stagnation_limit: Generations without best-fitness improvement before
            the stagnation response fires.  0 disables stagnation detection.
        stagnation_inject_count: Number of fresh random individuals injected
            (replacing the weakest members) when stagnation is detected.
        seed: Random seed for reproducibility.
    """

    population_size: int = 200
    max_generations: int = 500
    cx_prob: float = 0.7
    mut_prob: float = 0.2
    tournament_size: int = 5
    min_chromosome_length: int = 1
    max_chromosome_length: int = 100
    fitness_strategy: str = "misplaced_facelets"
    stagnation_limit: int = 50
    stagnation_inject_count: int = 10
    seed: int = 42
    progress_callback: Optional[Callable[[dict], None]] = field(
        default=None, repr=False, compare=False
    )


class GeneticSolver(AbstractSolver):
    """Solves puzzles using a genetic algorithm powered by DEAP.

    Each individual in the population is a variable-length chromosome — a list
    of integer indices into ``puzzle.legal_moves()``.  Fitness is computed by
    applying the chromosome to the scrambled puzzle and evaluating the
    resulting state with the configured fitness strategy.

    The solver terminates early when a chromosome decodes to a fully solved
    state.  If no solution is found within ``config.max_generations`` the best
    chromosome encountered is returned with ``solved=False``.

    Metadata keys in the returned :class:`~rubiks_solve.solvers.base.SolveResult`:
        ``fitness_history``: list of best fitness values, one per generation.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        config: GeneticConfig = GeneticConfig(),
    ) -> None:
        """Initialise the genetic solver.

        Args:
            puzzle_type: The concrete puzzle class to solve (e.g. ``Cube3x3``).
            config: Algorithm hyper-parameters.  Defaults to
                :class:`GeneticConfig` with sensible values.
        """
        super().__init__(puzzle_type, config)

        if config.fitness_strategy not in _FITNESS_STRATEGIES:
            raise ValueError(
                f"Unknown fitness_strategy {config.fitness_strategy!r}. "
                f"Choose from {list(_FITNESS_STRATEGIES)}"
            )
        self._fitness_fn = _FITNESS_STRATEGIES[config.fitness_strategy]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Run the genetic algorithm to solve *puzzle*.

        Args:
            puzzle: The scrambled puzzle instance to solve.

        Returns:
            A :class:`~rubiks_solve.solvers.base.SolveResult` containing the
            best move sequence found, whether it is a full solution, the number
            of generations run, and per-generation fitness history.
        """
        cfg: GeneticConfig = self.config
        start_time = time.perf_counter()

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        legal_moves: list[Move] = puzzle.legal_moves()
        n_moves = len(legal_moves)

        if n_moves == 0:
            return SolveResult(
                solved=puzzle.is_solved,
                moves=[],
                solve_time_seconds=time.perf_counter() - start_time,
                iterations=0,
                metadata={"fitness_history": []},
            )

        toolbox, make_individual = self._build_toolbox(puzzle, legal_moves, n_moves, cfg)

        # Initialise population
        population: list[Any] = toolbox.population(n=cfg.population_size)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        fitness_best_history: list[float] = []
        fitness_mean_history: list[float] = []
        control_fitness_history: list[float] = []
        stagnation_events: list[int] = []  # generation indices where injection fired

        best_chromosome: list[int] = list(population[0])
        best_fitness: float = population[0].fitness.values[0]
        stagnation_count: int = 0
        solved = False

        for gen_idx in range(cfg.max_generations):
            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < cfg.cx_prob:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Mutation
            for ind in offspring:
                if random.random() < cfg.mut_prob:
                    toolbox.mutate(ind)
                    del ind.fitness.values

            # Evaluate individuals with invalid fitness
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

            # --- Per-generation metrics ---
            all_fits = [ind.fitness.values[0] for ind in population]
            gen_fitness = max(all_fits)
            gen_mean = float(np.mean(all_fits))
            fitness_best_history.append(gen_fitness)
            fitness_mean_history.append(gen_mean)

            # Control individual: one fresh random chromosome evaluated this gen
            control_ind = make_individual()
            control_fitness_history.append(toolbox.evaluate(control_ind)[0])

            # --- Stagnation detection & injection ---
            if gen_fitness > best_fitness:
                best_fitness = gen_fitness
                best_chromosome = list(
                    max(population, key=lambda ind: ind.fitness.values[0])
                )
                stagnation_count = 0
            else:
                stagnation_count += 1

            if cfg.stagnation_limit > 0 and stagnation_count >= cfg.stagnation_limit:
                # Replace the weakest members with fresh random individuals
                sorted_pop = sorted(
                    range(len(population)),
                    key=lambda i: population[i].fitness.values[0],
                )
                for slot in sorted_pop[: cfg.stagnation_inject_count]:
                    fresh = make_individual()
                    fresh.fitness.values = toolbox.evaluate(fresh)
                    population[slot] = fresh
                stagnation_events.append(gen_idx)
                stagnation_count = 0
                self._logger.debug(
                    "Stagnation injection at generation %d — injected %d individuals",
                    gen_idx,
                    cfg.stagnation_inject_count,
                )

            # --- Progress callback ---
            if cfg.progress_callback is not None:
                cfg.progress_callback({
                    "type": "genetic",
                    "generation": gen_idx + 1,
                    "max_generations": cfg.max_generations,
                    "best_fitness": gen_fitness,
                    "mean_fitness": gen_mean,
                    "elapsed": time.perf_counter() - start_time,
                })

            # Early exit on solution
            result_puzzle = puzzle.apply_moves(
                self._decode_chromosome(best_chromosome, legal_moves)
            )
            if result_puzzle.is_solved:
                solved = True
                break

        solution_moves = self._decode_chromosome(best_chromosome, legal_moves)
        elapsed = time.perf_counter() - start_time

        # Final verification in case the early-exit was not triggered
        if not solved:
            solved = puzzle.apply_moves(solution_moves).is_solved

        return SolveResult(
            solved=solved,
            moves=solution_moves,
            solve_time_seconds=elapsed,
            iterations=len(fitness_best_history),
            metadata={
                "fitness_history": fitness_best_history,       # backward compat
                "fitness_best_history": fitness_best_history,
                "fitness_mean_history": fitness_mean_history,
                "control_fitness_history": control_fitness_history,
                "stagnation_events": stagnation_events,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_toolbox(
        self,
        puzzle: AbstractPuzzle,
        legal_moves: list[Move],
        n_moves: int,
        cfg: GeneticConfig,
    ) -> tuple[base.Toolbox, Any]:
        """Construct and return a configured DEAP toolbox and individual factory.

        Args:
            puzzle: The scrambled puzzle instance used by the fitness function.
            legal_moves: Ordered list of legal moves for index mapping.
            n_moves: ``len(legal_moves)``; passed to the mutation operator.
            cfg: Algorithm configuration.

        Returns:
            Tuple of (toolbox, make_individual) where make_individual is a
            callable that creates a fresh random Individual.
        """
        toolbox = base.Toolbox()

        # Random gene generator
        toolbox.register("attr_move", random.randrange, n_moves)

        # Individual factory: variable-length list of move indices
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_move,
            n=cfg.min_chromosome_length,
        )

        # Override individual init to use random length in [min, max]
        def _make_individual() -> creator.Individual:
            length = random.randint(
                cfg.min_chromosome_length, cfg.max_chromosome_length
            )
            genes = [random.randrange(n_moves) for _ in range(length)]
            return creator.Individual(genes)

        toolbox.register("individual", _make_individual)
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.individual
        )

        fitness_fn = self._fitness_fn

        def _evaluate(individual: list[int]) -> tuple[float]:
            return (fitness_fn(individual, puzzle, legal_moves),)

        toolbox.register("evaluate", _evaluate)
        toolbox.register("mate", cx_single_point)
        toolbox.register(
            "mutate", mut_random_move, n_moves=n_moves, indpb=0.2
        )
        toolbox.register(
            "select",
            sel_tournament_variable,
            tournsize=cfg.tournament_size,
        )

        return toolbox, _make_individual

    def _decode_chromosome(
        self, chromosome: list[int], legal_moves: list[Move]
    ) -> list[Move]:
        """Convert a list of move indices to a list of :class:`~rubiks_solve.core.base.Move` objects.

        Args:
            chromosome: Sequence of indices into *legal_moves*.
            legal_moves: Ordered list of legal moves for the puzzle type.

        Returns:
            Corresponding list of :class:`~rubiks_solve.core.base.Move` instances.
        """
        return [legal_moves[idx] for idx in chromosome]
