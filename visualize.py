"""
Generate (unoptimized) asciinema replay files for the blog post.
"""
import time
from typing import List, TextIO

from solver import ship_fits, count_occurances

FILE: TextIO = open(r"output.cast", "w+", encoding="UTF8")
TIMESTAMP: int = 0


def write(output: str):
    """Write a line to the cast file"""
    global TIMESTAMP

    output = output.replace("\n", "\\n")

    FILE.write(f'[{TIMESTAMP}, "o", "{output}"]\n')

    TIMESTAMP += 0.5


def print_board(board):
    """Write a board to the cast file"""
    board_str = ""
    for row in board:
        for cell in row:
            board_str += print_pos(cell)
        board_str = board_str.strip()
        # board_str += "\n"

    # print(board_str, end="")
    write(board_str)


def print_num_board(board):
    """Write a zero-padded board to the cast file"""
    board_str = ""
    for row in board:
        for cell in row:
            board_str += str(cell).zfill(5) + " "
        board_str = board_str.strip()
        # board_str += "\n"

    # print(board_str, end="")
    write(board_str)


def print_pos(cell: int) -> str:
    """Return the symbol associated with the cell type"""
    if cell == 0:
        return ". "
    if cell == 1:
        return "- "
    if cell == 2:
        return "X "

    raise NotImplementedError


def solo_ship_board(i, j, ship, horizontal) -> List[List[int]]:
    """Board containing single ship"""
    board = [[1 for _ in range(10)] for _ in range(10)]
    if horizontal:
        for k in range(j, j + ship):
            board[i][k] = 2
    else:
        for k in range(i, i + ship):
            board[k][j] = 2

    return board


def enumerate_positions(ships, board):
    """Enumerate over every possible ship position"""

    for i in range(len(board)):
        for j in range(len(board[i])):
            for ship, _ in ships.items():
                i_fits, j_fits = ship_fits(i, j, ship, board)

                if i_fits:
                    print_board(solo_ship_board(i, j, ship, False))
                if j_fits:
                    print_board(solo_ship_board(i, j, ship, True))


def enumerate_num_positions(ships, board):
    """Enumerate over every possible position and add the ship occurrences at each cell"""
    res = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]

    for i in range(len(board)):
        for j in range(len(board[i])):
            for ship, count in ships.items():
                i_fits, j_fits = ship_fits(i, j, ship, board)

                if i_fits:
                    for k in range(i, i + ship):
                        res[k][j] += count
                    print_num_board(res)
                    time.sleep(0.01)
                if j_fits:
                    for k in range(j, j + ship):
                        res[i][k] += count
                    print_num_board(res)
                    time.sleep(0.01)


def sum_row(i, j, size, source):
    """Add all the cells in a row/col"""
    if i < 0 or i >= len(source):
        return 0
    if j < 0 or j + size > len(source):
        return 0

    if j == 0:
        size += 1
    else:
        j -= 1
        size += 2

    return sum(source[i][j : j + size])


def position_sum(i, j, ship, vert, source):
    """Sum up the cells surrounding a position"""
    psum = 0

    if vert:
        for k in range(ship + 2):
            psum += sum_row(i, j + k - 1, 1, source)
    else:
        for k in range(3):
            psum += sum_row(i, j + k - 1, ship, source)

    return psum


def enumerate_smart_sum(ships, board):
    """
    Enumerate over every possible position and add the
    sum of this cell plus surrounding ship occurrences
    """
    result = count_occurances(board, ships)

    res = [[0 for _ in range(10)] for _ in range(10)]

    for i in range(len(board)):
        for j in range(len(board)):
            for ship, _ in ships.items():
                i_fits, j_fits = ship_fits(i, j, ship, board)

                if i_fits:
                    count = position_sum(i, j, ship, True, result)
                    print("i", ship, i, j, count)
                    print(res)
                    for k in range(i, i + ship):
                        res[k][j] += count
                    print_num_board(res)
                if j_fits:
                    count = position_sum(i, j, ship, False, result)
                    print(ship, i, j, count)
                    print(res)
                    for k in range(j, j + ship):
                        res[i][k] += count
                    print_num_board(res)

    print(result)
    print(res)


def visualize():
    """Write an enumeration to a cast file"""
    FILE.write(
        '{"version": 2, "width": 29, "height": 10, "timestamp": 0, '
        '"env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"}}\n'
    )
    print("Hi!")
    board = [[0 for _ in range(10)] for _ in range(10)]

    enumerate_smart_sum({4: 1, 3: 2, 2: 3, 1: 4}, board)


if __name__ == "__main__":
    visualize()
