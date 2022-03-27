"""
Messy incomplete implementation of strategy described in: https://www.nulliq.dev/posts/battleship/
"""
import math
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from scipy import signal
import seaborn as sns
import mplcursors

SHIPS: Dict[int, int] = {4: 1, 3: 2, 2: 3, 1: 4}

GRID: np.array = np.array([[0 for _ in range(10)] for _ in range(10)])

PLOT: matplotlib.pyplot = plt


def main():
    """Program entry point"""
    # It's recommended to not enable exhaustive search until mid to late game.
    heatdata, _, _ = solve(exhaustive=True)
    init_plot(heatdata)


def ship_fits(i, j, ship, board=None) -> (int, int):
    """Does this size ship fit at these coordinates?"""
    if board is None:
        board = GRID

    i_fits = True
    j_fits = True

    if i + ship <= len(board):
        for k in range(i, i + ship):
            if board[k][j] != 0:
                i_fits = False
    else:
        i_fits = False

    if j + ship <= len(board[i]):
        for k in range(j, j + ship):
            if board[i][k] != 0:
                j_fits = False
    else:
        j_fits = False

    if ship == 1:
        j_fits = False

    return i_fits, j_fits


def info_sum(
    ship: int, i: int, j: int, singles: np.array, spine: np.array, vert: bool
) -> int:
    """Add together the surrounding cells for the given ship position"""
    last_i = i
    last_j = j

    if vert:
        last_j += ship - 1
    else:
        last_i += ship - 1

    if ship == 1:
        return singles[i][j]
    if ship == 2:
        # Add up two ends of ship and delete the double-counted center spines
        return (
            singles[i][j]
            + singles[last_i][last_j]
            - spine[math.floor((i + last_i) / 2)][math.floor((j + last_j) / 2)]
            - spine[math.ceil((i + last_i) / 2)][math.ceil((j + last_j) / 2)]
        )
    if ship == 3:
        # Add up two ends of ship and delete the double-counted center spine
        return (
            singles[i][j]
            + singles[last_i][last_j]
            - spine[(i + last_i) // 2][(j + last_j) // 2]
        )
    if ship == 4:
        # Add up two ends of ship
        return singles[i][j] + singles[last_i][last_j]

    # Unsupported ship len
    raise NotImplementedError


def count_occurances(grid: np.array, ships: Dict[int, int]) -> List[List[int]]:
    """Count how many different ships can fit into each cell on the grid."""
    res = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            for ship, count in ships.items():
                fits = ship_fits(i, j, ship)
                if fits[0]:  # fits with horizontal orientation
                    for k in range(i, i + ship):
                        res[k][j] += count
                if fits[1]:  # fits vertically
                    for k in range(j, j + ship):
                        res[i][k] += count

    return res


def place_ship(i: int, j: int, ship: int, grid: np.array, vert: bool) -> np.array:
    """Increment the cells that could not contain another ship if the ship occupied the given coords."""
    # Effective values
    e_i = i - 1
    e_j = j - 1
    if vert:
        e_i_delta = ship + 2
        e_j_delta = 3
    else:
        e_i_delta = 3
        e_j_delta = ship + 2

    if e_i < 0:
        e_i_delta -= 1
        e_i += 1
    if e_i + e_i_delta > len(grid):
        e_i_delta -= 1

    if e_j < 0:
        e_j += 1
        e_j_delta -= 1
    if e_j + e_j_delta > len(grid[i]):
        e_j_delta -= 1

    add_grid = np.ones((e_i_delta, e_j_delta))

    add_grid = np.pad(add_grid, ((e_i, len(grid)-e_i_delta-e_i), (e_j, len(grid[i])-e_j_delta-e_j)), 'constant')

    return np.add(grid, add_grid)


def permutate_board(GRID: np.array, SHIPS: dict):
    """Exhaustively count valid board configurations - See github issue #1"""
    all_prob = np.zeros_like(GRID)

    # Base case
    if len(SHIPS.keys()) == 0:
        return all_prob

    # Place largest ship
    ship = max(SHIPS.keys())
    ships = SHIPS.copy()
    if ships[ship] == 1:
        ships.pop(ship)
    else:
        ships[ship] -= 1

    for i in range(len(GRID)):
        for j in range(len(GRID[i])):
            # Place ship
            vert, hor = ship_fits(i, j, ship, GRID)

            vert_prob = None
            if vert:
                grid = GRID.copy()
                grid = place_ship(i, j, ship, grid, vert=True)

                # Recursively call self
                vert_prob = permutate_board(grid, ships)
                if vert_prob is not None:
                    for k in range(i, i + ship):
                        vert_prob[k][j] += 1

            hor_prob = None
            if hor:
                grid = GRID.copy()

                grid = place_ship(i, j, ship, grid, vert=False)
                # Recursively call self
                hor_prob = permutate_board(grid, ships)
                if hor_prob is not None:
                    for k in range(j, j + ship):
                        hor_prob[i][k] += 1

            # Sum up probability counts
            if vert_prob is not None and hor_prob is not None:
                res = np.add(vert_prob, hor_prob)
                all_prob = np.add(all_prob, res)
            elif vert_prob is not None:
                all_prob = np.add(all_prob, vert_prob)
            elif hor_prob is not None:
                all_prob = np.add(all_prob, hor_prob)

    # If there weren't any valid placements, return None
    if all_prob.sum() == 0:
        return None

    return all_prob


def solve(exhaustive=False) -> Tuple[np.array, np.array, np.array]:
    """Generate the heatmap data using the methods described in the blogpost"""
    if exhaustive:
        prob = permutate_board(GRID, SHIPS)
    else:
        prob = count_occurances(GRID, SHIPS)

    matrix = np.array(prob)

    size = 3

    kernel = np.ones((size, size))
    result = signal.convolve(matrix, kernel, method="direct").astype(int)
    singles = result[
        (size - 1) // 2 : -(size - 1) // 2, (size - 1) // 2 : -(size - 1) // 2
    ]
    singles = np.where(matrix != 0, singles, 0)

    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    result = signal.convolve(matrix, kernel, method="direct").astype(int)
    vert = result[
        (size - 1) // 2 : -(size - 1) // 2, (size - 1) // 2 : -(size - 1) // 2
    ]
    vert = np.where(matrix != 0, vert, 0)

    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    result = signal.convolve(matrix, kernel, method="direct").astype(int)
    hor = result[(size - 1) // 2 : -(size - 1) // 2, (size - 1) // 2 : -(size - 1) // 2]
    hor = np.where(matrix != 0, hor, 0)

    print("vert/hor:")
    print(vert)
    print(hor)

    weighted_info = np.array(
        [[0 for i in range(len(GRID[0]))] for j in range(len(GRID))]
    )
    total_lens = np.array([[0 for i in range(len(GRID[0]))] for j in range(len(GRID))])

    for i in range(len(GRID)):
        for j in range(len(GRID[i])):
            for ship, count in SHIPS.items():
                fits = ship_fits(i, j, ship)
                if fits[0]:  # fits with horizontal orientation
                    total_lens[i][j] += ship * count
                    wsum = info_sum(ship, i, j, singles, hor, vert=False)
                    for k in range(i, i + ship):
                        weighted_info[k][j] += wsum * count
                if fits[1]:  # fits vertically
                    total_lens[i][j] += ship * count
                    wsum = info_sum(ship, i, j, singles, vert, vert=True)
                    for k in range(j, j + ship):
                        weighted_info[i][k] += wsum * count

    # If we used the exhaustive search, we know for a fact that the zeros cannot contain a ship.
    if exhaustive:
        weighted_info = np.where(matrix != 0, weighted_info, 0)
        vert = np.where(matrix != 0, vert, 0)
        hor = np.where(matrix != 0, hor, 0)

    return weighted_info, vert, hor


def init_plot(data):
    """Generate a matplotlib seaborn heatmap"""

    cdict = {
        "red": [(0.0, 0.129, 0.129), (1.0, 0.933, 1.0)],
        "green": [(0.0, 0.125, 0.125), (1.0, 0.447, 1.0)],
        "blue": [(0.0, 0.173, 0.173), (1.0, 0.945, 1.0)],
    }

    cmap = LinearSegmentedColormap("test", cdict)

    ax = sns.heatmap(data, linewidth=0.5, cmap=cmap)

    max_coords = data.argmax()

    dummy_image = ax.imshow(data, zorder=-1, aspect="auto")
    cursor = mplcursors.cursor(dummy_image, hover=False)
    cursor.connect("add", toggle_square)

    plt.title(f"Hit the red ({max_coords % len(GRID)}, {max_coords // len(GRID[0])})")

    vaxes = plt.axes([0.81, 0.000001, 0.1, 0.075])
    bvert = Button(vaxes, "Vert", color="yellow")
    bvert.on_clicked(vert_plot)

    haxes = plt.axes([0.70, 0.000001, 0.1, 0.075])
    bhor = Button(haxes, "Horiz", color="yellow")
    bhor.on_clicked(hor_plot)

    plt.show()


def search_plot(press):
    """
    Broken, haven't gotten around to figuring out how to update matplotlib seaborn figures
    without closing the window.
    """
    update_plot(solve()[0])


def vert_plot(press):
    """
    Broken, haven't gotten around to figuring out how to update matplotlib seaborn figures
    without closing the window.
    """
    update_plot(solve()[1])


def hor_plot(press):
    """
    Broken, haven't gotten around to figuring out how to update matplotlib seaborn figures
    without closing the window.
    """
    update_plot(solve()[2])


def update_plot(data: np.array):
    """It's a hack. Definitely a better way to do this, but hey, this works."""
    PLOT.close()

    init_plot(data)


def toggle_data(cell_x: int, cell_y: int):
    """Toggle a grid cell"""
    if GRID[cell_y][cell_x] == 0:
        GRID[cell_y][cell_x] = 1
    elif GRID[cell_y][cell_x] == 1:
        GRID[cell_y][cell_x] = 2
    elif GRID[cell_y][cell_x] == 2:
        GRID[cell_y][cell_x] = 0

    print(repr(GRID))


def toggle_square(press):
    """Translate a click event and toggle a square"""
    cell_x = int(press.target[0])
    cell_y = int(press.target[1])

    toggle_data(cell_x, cell_y)
    data = solve()[0]
    update_plot(data=data)


if __name__ == "__main__":
    main()
