import random
import pickle
import os
import ast

BOARD_SIZE = 4

# =========================================================
# Orbit 규칙
# =========================================================

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
# 패배 상태 저장 (pickle 사용)
# =========================================================

LOSS_FILE = "losing_states.pkl"

# 시작 시 로드
if os.path.exists(LOSS_FILE):

    try:
        with open(LOSS_FILE, "rb") as f:
            KNOWN_LOSSES = pickle.load(f)

    except:
        KNOWN_LOSSES = set()

else:
    KNOWN_LOSSES = set()


def save_losses():

    with open(LOSS_FILE, "wb") as f:
        pickle.dump(KNOWN_LOSSES, f)


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

def legal_moves(board):

    moves = []

    for i in range(16):

        if board[i] == '':
            moves.append(str(i))

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

        # 내 라인
        if mine > 0 and theirs == 0:
            score += mine * mine

        # 상대 라인
        if theirs > 0 and mine == 0:
            score -= theirs * theirs

    return score


# =========================================================
# Minimax
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

    # 이미 졌던 상태면 강한 패널티
    if state_key in KNOWN_LOSSES:
        return -9999

    moves = legal_moves(board)

    if maximizing:

        best = -999999

        for mv in moves:

            next_board = apply_move(board, mv, turn_color)

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

        for mv in moves:

            next_board = apply_move(board, mv, turn_color)

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

def move(game_state):

    """
    입력 예시:

    "['', '', '', '', '', '', '', '',
      '', '', '', '', '', '', '', '']/4/4/w"

    형식:
    board / black_left / white_left / my_color
    """

    # -----------------------------------------------------
    # 문자열 파싱
    # -----------------------------------------------------

    parts = game_state.split('/')

    board_str = parts[0]

    black_left = int(parts[1])
    white_left = int(parts[2])

    my_color = parts[3]

    # 문자열 리스트 -> 실제 리스트
    board = ast.literal_eval(board_str)

    # -----------------------------------------------------
    # 탐색
    # -----------------------------------------------------

    best_score = -999999
    best_move = None

    moves = legal_moves(board)

    random.shuffle(moves)

    for mv in moves:

        next_board = apply_move(board, mv, my_color)

        key = serialize(next_board, opponent(my_color))

        # 이미 패배했던 상태면 제외
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

    # 전부 패배 루트면 랜덤 선택
    if best_move is None:
        best_move = random.choice(moves)

    return best_move


# =========================================================
# 학습
# =========================================================

def remember_loss(history, loser_color):

    updated = False

    for board, turn in history:

        if turn == loser_color:

            key = serialize(board, turn)

            if key not in KNOWN_LOSSES:

                KNOWN_LOSSES.add(key)
                updated = True

    # 새 데이터가 있을 때만 저장
    if updated:
        save_losses()
