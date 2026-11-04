"""Memory-1 strategies encoded as 5-bit lookup tables.

Bit layout (0 = Cooperate, 1 = Defect):
    index 0 -> first action
    index 1 -> action after CC   (own action first, opponent action second)
    index 2 -> action after CD
    index 3 -> action after DC
    index 4 -> action after DD
"""

Strategy = tuple[int, ...]

STRATEGY_LENGTH: int = 5
N_STRATEGIES: int = 2 ** STRATEGY_LENGTH  # 32

# Bits:first, CC, CD, DC, DD
NAMED_STRATEGIES: dict[str, Strategy] = {
    "ALLC":   (0, 0, 0, 0, 0),  # always cooperate
    "ALLD":   (1, 1, 1, 1, 1),  # always defect
    "TFT":    (0, 0, 1, 0, 1),  # start C, then copy the opponent's last action
    "STFT":   (1, 0, 1, 0, 1),  # suspicious TFT: like TFT but starts with D
    "PAVLOV": (0, 0, 1, 1, 0),  # win-stay (CC, DC), lose-shift (CD, DD)
    "GRIM":   (0, 0, 1, 1, 1),  # cooperate until any defection, then defect forever
}


def first_action(strategy: Strategy) -> int:
    """Action played in the first round."""
    return strategy[0]


def next_action(strategy: Strategy, own_last: int, opponent_last: int) -> int:
    """Action to play given the previous round, seen from this player's side."""
    # CC -> 1, CD -> 2, DC -> 3, DD -> 4
    outcome_index = 1 + 2 * own_last + opponent_last
    return strategy[outcome_index]


def to_int(strategy: Strategy) -> int:
    """Convert a 5-bit strategy into an integer 0..31 (first bit is the most significant)."""
    number = 0
    for bit in strategy:
        number = 2 * number + bit
    return number


def from_int(number: int) -> Strategy:
    """Convert an integer 0..31 back into its 5-bit strategy."""
    if not 0 <= number < N_STRATEGIES:
        raise ValueError(f"Strategy number must be in 0..{N_STRATEGIES - 1}, got {number}")
    bits = []
    for position in range(STRATEGY_LENGTH - 1, -1, -1):
        bits.append((number >> position) & 1)
    return tuple(bits)


def strategy_name(strategy: Strategy) -> str:
    """Return the classical name if it has one, otherwise a generic label like 'S13'."""
    for name, bits in NAMED_STRATEGIES.items():
        if tuple(strategy) == bits:
            return name
    return f"S{to_int(strategy):02d}"
