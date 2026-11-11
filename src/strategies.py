# strategies are bitstrings: tuple of binary integers
def strategy_length(memory: int):
    return memory + 4 ** memory # memory 1 => length 5; memory 3 => length 18


# Memory 1. Bits: first, CC, CD, DC, DD
NAMED_STRATEGIES = {
    "ALLC":   (0, 0, 0, 0, 0),  # always cooperate
    "ALLD":   (1, 1, 1, 1, 1),  # always defect
    "TFT":    (0, 0, 1, 0, 1),  # start C, then copy the opponent's last action
    "PAVLOV": (0, 0, 1, 1, 0),  # win-stay (CC, DC), lose-shift (CD, DD)
    "GRIM":   (0, 0, 1, 1, 1),  # cooperate until any defection, then defect forever (unless an error makes it cooperate and oponent also cooperates, returning to initial cooperation state).
} # Memory 2 strategies are not given names (too many configurations)


def choose_action(strategy, own_history, opponent_history):
    if len(strategy) == strategy_length(1):
        memory = 1
    elif len(strategy) == strategy_length(2):
        memory = 2
    else:
        raise ValueError("Invalid length for the strategy!")

    round_number = len(own_history)

    if round_number < memory: # initial moves
        return strategy[round_number] 

    outcome = 0
    for own, opponent in zip(own_history[-memory:], opponent_history[-memory:]): 
        outcome = 4 * outcome + 2 * own + opponent # find the index in the configuration
    return strategy[memory + outcome]


def to_int(strategy): # conversion of bitstring into string (0-2^memory-1)
    number = 0
    for bit in strategy:
        number = 2 * number + bit
    return number


def from_int(number, length):
    if not 0 <= number < 2 ** length:
        raise ValueError(f"Strategy number must be in 0..{2 ** length - 1}, got {number}")
    return tuple(int(bit) for bit in format(number, f"0{length}b")) # turn into bit format


def strategy_name(strategy):
    for name, bits in NAMED_STRATEGIES.items():
        if tuple(strategy) == bits:
            return name
    return "OTHER"

