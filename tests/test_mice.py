import pytest
import mice
import numpy as np
from collections import namedtuple

#==== testing mice.lattices_generator ====#
lattices_generator_input = namedtuple('input', ['R', 'num_frames', 'num_boxes', 'sizes'])
input1 = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=4, sizes=(4,4,4))
input2 = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=7, sizes=(5,5,5))
input3 = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=8, sizes=(6,6,6))
@pytest.mark.parametrize("test_input,expected", [
    (input1, list),
    (input2, list),
    (input3, list),
])
def test_lattices_generator(test_input, expected):
    R, num_frames, num_boxes, sizes = test_input
    assert type(mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)) is expected

#==== testing mice.lattices_splitter ====#
lattices_generator_input = namedtuple('input', ['R', 'num_frames', 'num_boxes', 'sizes'])
R, num_frames, num_boxes, sizes = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=4, sizes=(4,4,4))
input1 = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
R, num_frames, num_boxes, sizes = lattices_generator_input(R=np.random.RandomState(seed=1), num_frames=1000, num_boxes=7, sizes=(6,6,6))
input2 = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
R, num_frames, num_boxes, sizes = lattices_generator_input(R=np.random.RandomState(seed=2), num_frames=2000, num_boxes=8, sizes=(6,6,6))
input3 = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
@pytest.mark.parametrize("test_input,expected", [
    (input1, tuple),
    (input2, tuple),
    (input3, tuple),
])
def test_lattice_splitter(test_input, expected):
    assert type(mice.lattice_splitter(lattices=test_input, axis=0)) is expected

# ==== not implemented yet ==== #
@pytest.mark.skip(reason="this is not implemented yet")
def test_not_implemented():
    return None

if __name__ == '__main__':
    # === if we want to use the debugger ==== #
    lattices_generator_input = namedtuple('input', ['R', 'num_frames', 'num_boxes', 'sizes'])
    R, num_frames, num_boxes, sizes = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=4, sizes=(4,4,4))
    input1 = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
    R, num_frames, num_boxes, sizes = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=7, sizes=(5,5,5))
    input2 = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
    R, num_frames, num_boxes, sizes = lattices_generator_input(R=np.random.RandomState(seed=0), num_frames=2000, num_boxes=8, sizes=(6,6,6))
    input3 = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
    test_lattice_splitter(input2, tuple)