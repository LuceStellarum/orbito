from collections import defaultdict
import random
import json
import os

BOARD_SIZE = 4

# =========================================================
# Orbit 이동 규칙
# =========================================================
#
# 인덱스:
#
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
#
# 실제 Orbito 규칙에 맞게 수정 가능
#

ORBIT_MAP = {
    0: 1,
    1: 2,
    2: 3,
    3: 7,
    7: 11,
    11: 15,
    15: 14,
    14: 13,
    13: 12,
    12: 8,
    8: 4,
    4: 0,

    5: 6,
    6: 10,
    10: 9,
    9: 5
}

WIN_LINES = [
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11],
    [12,13,14,15],

    [0,4,8,12],
    [1,5,9,13],
    [2,6,10,14],
    [3,7,11,15],

    [0,5,10,15],
    [3,6,9,12]
]


# =========================================================
# 학습 데이터 저장
# =========================================================

LOSS_FILE = "losing_states.json"

if os.path.exists(LOSS_FILE):
    with open(LOSS_FILE, "r") as f:
        KNOWN_LOSSES = set(json.load(f))
else:
    KNOWN_LOSSES = set()


def save_losses():
    with open(LOSS_FILE, "w") as f:
        json.dump(list(KNOWN_LOSSES), f)


# =========================================================
# 유틸
# =========================================================

def serialize(board, turn):
    return str(board) + "|" + turn


def opponent(color):
    return 'b' if color == 'w' else 'w'


# =========================================================
# 승리 판정
# =========================================================

def check_winner(board):

    for line in WIN_LINES:

        stones = [board[i] for i in line]

        if stones == ['b'] * 4:
            return 'b'

        if stones == ['w'] * 4:
            return 'w'

    return None


# =========================================================
# Orbit 회전
# =========================================================

def rotate_board(board):

    new_board = board[:]

    temp = board[:]

    for src, dst in ORBIT_MAP.items():
        new_board[dst] = temp[src]

    return new_board


# =========================================================
# 가능한 수 생성
# =========================================================

def legal_moves(board, my_stones):

    moves = []

    # 그냥 착수
    for i in range(16):
        if board[i] == '':
            moves.append(str(i))

    # 밀기 기능 추가 가능
    # 예:
    #
    # "5>6/10"
    #
    # 의미:
    # 5의 돌을 6으로 이동 후
    # 10에 새 돌 착수

    return moves


# =========================================================
# 수 적용
# =========================================================

def apply_move(board, move, color):

    board = board[:]

    # 단순 착수
    if '/' not in move and '>' not in move:

        pos = int(move)
        board[pos] = color

    # 이동 + 착수
    else:

        move_part, place_part = move.split('/')

        src, dst = move_part.split('>')

        src = int(src)
        dst = int(dst)
        place_part = int(place_part)

        board[dst] = board[src]
        board[src] = ''

        board[place_part] = color

    # orbit 회전
    board = rotate_board(board)

    return board


# =========================================================
# 평가 함수
# =========================================================

def evaluate(board, my_color):

    enemy = opponent(my_color)

    score = 0

    for line in WIN_LINES:

        mine = 0
        theirs = 0

        for i in line:

            if board[i] == my_color:
                mine += 1

            elif board[i] == enemy:
                theirs += 1

        if mine > 0 and theirs == 0:
            score += mine * mine

        if theirs > 0 and mine == 0:
            score -= theirs * theirs

    return score


# =========================================================
# minimax + 패배 회피
# =========================================================

MAX_DEPTH = 3


def minimax(board, depth, maximizing, my_color, turn_color):

    winner = check_winner(board)

    if winner == my_color:
        return 10000

    if winner == opponent(my_color):
        return -10000

    if depth == 0:
        return evaluate(board, my_color)

    state_key = serialize(board, turn_color)

    # 이미 졌던 상태 회피
    if state_key in KNOWN_LOSSES:
        return -9999

    moves = legal_moves(board, 0)

    if maximizing:

        best = -999999

        for move in moves:

            next_board = apply_move(board, move, turn_color)

            score = minimax(
                next_board,
                depth - 1,
                False,
                my_color,
                opponent(turn_color)
            )

            best = max(best, score)

        return best

    else:

        best = 999999

        for move in moves:

            next_board = apply_move(board, move, turn_color)

            score = minimax(
                next_board,
                depth - 1,
                True,
                my_color,
                opponent(turn_color)
            )

            best = min(best, score)

        return best


# =========================================================
# 메인 move 함수
# =========================================================

def move(board, black_left, white_left, my_color):

    """
    board:
    ['', '', 'b', 'w', ...]

    return:
    "5"
    또는
    "3>7/10"
    """

    best_score = -999999
    best_move = None

    moves = legal_moves(board, 0)

    random.shuffle(moves)

    for mv in moves:

        next_board = apply_move(board, mv, my_color)

        key = serialize(next_board, opponent(my_color))

        # 이미 패배한 상태 회피
        if key in KNOWN_LOSSES:
            continue

        score = minimax(
            next_board,
            MAX_DEPTH,
            False,
            my_color,
            opponent(my_color)
        )

        if score > best_score:
            best_score = score
            best_move = mv

    # 모든 수가 패배 상태면 랜덤
    if best_move is None:
        best_move = random.choice(moves)

    return best_move


# =========================================================
# 학습용 함수
# =========================================================

def remember_loss(history, loser_color):

    """
    history:
    [(board, turn), ...]
    """

    for board, turn in history:

        if turn == loser_color:

            key = serialize(board, turn)

            KNOWN_LOSSES.add(key)

    save_losses()
