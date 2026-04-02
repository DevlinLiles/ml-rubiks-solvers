import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def solved_2x2():
    from rubiks_solve.core.cube_2x2 import Cube2x2
    return Cube2x2.solved_state()


@pytest.fixture
def solved_3x3():
    from rubiks_solve.core.cube_3x3 import Cube3x3
    return Cube3x3.solved_state()


@pytest.fixture
def solved_4x4():
    from rubiks_solve.core.cube_4x4 import Cube4x4
    return Cube4x4.solved_state()


@pytest.fixture
def solved_5x5():
    from rubiks_solve.core.cube_5x5 import Cube5x5
    return Cube5x5.solved_state()


@pytest.fixture
def solved_megaminx():
    from rubiks_solve.core.megaminx import Megaminx
    return Megaminx.solved_state()


@pytest.fixture
def scrambled_3x3_5(solved_3x3, rng):
    return solved_3x3.scramble(5, rng)


@pytest.fixture
def scrambled_3x3_20(solved_3x3, rng):
    return solved_3x3.scramble(20, rng)
