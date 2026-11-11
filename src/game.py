import numpy as np
from strategies import choose_action, strategy_length

# Cooperate: 0 = 0
# Defect: D = 1


def flip(action):
    return 1 - action 


class PayoffMatrix:
    def __init__(self):
        self.T = 5 
        self.R = 3
        self.P = 1
        self.S = 0
        
    def scores(self, a: int, b: int) -> tuple[float, float]:
        # table[my_action][other_action] = my payoff
        table = [[self.R, self.S],
                 [self.T, self.P]]
        return table[a][b], table[b][a]


def apply_noise(action, p_noise, rng):
    if rng.random() < p_noise:
        return flip(action)
    return action


def play_match(player, opponent, n_rounds, p_noise, payoffs, rng):
    if payoffs is None:
        payoffs = PayoffMatrix()
    if rng is None:
        rng = np.random.default_rng()

    player_score = 0
    opponent_score = 0
    player_C = 0
    opponent_C = 0 
    # Actions actually played in each round (mistakes included)
    player_history = []
    opponent_history = []

    for _ in range(n_rounds):
        player_intended = choose_action(player, player_history, opponent_history)
        opponent_intended = choose_action(opponent, opponent_history, player_history)

        player_action = apply_noise(player_intended, p_noise, rng)
        opponent_action = apply_noise(opponent_intended, p_noise, rng)

        player_payoff, opponent_payoff = payoffs.scores(player_action, opponent_action)
        player_score += player_payoff
        opponent_score += opponent_payoff

        # count cooperation
        if player_action == 0:
            player_C += 1
        if opponent_action == 0:
            opponent_C += 1
            
        player_history.append(player_action)
        opponent_history.append(opponent_action)

    return player_score, opponent_score, player_C, opponent_C


def play_many_matches(players, opponents, n_rounds, p_noise, rng, payoffs):
    if payoffs is None:
        payoffs = PayoffMatrix()

    n_matches = players.shape[0] # [n_matches, length]
    if players.shape[1] == strategy_length(1):
        memory = 1
    elif players.shape[1] == strategy_length(2):
        memory = 2
    else:
        raise ValueError("Invalid length for the strategy!")

    n_outcomes = 4 ** memory  # 4 for memory 1, 16 for memory 2 

    # payoff_table[my_action, other_action] = my payoff (0 = 0, 1 = D)
    payoff_table = np.array([[payoffs.R, payoffs.S], [payoffs.T, payoffs.P]])

    # Score over rounds
    player_scores = np.zeros(n_matches) # [n_matches,]
    opponent_scores = np.zeros(n_matches) # [n_matches,]
    
    # Cooperation over rounds
    player_C = np.zeros(n_matches) 
    opponent_C = np.zeros(n_matches) 

    # Outcome number of the last memory rounds, seen from each side (same number as in choose_action)
    player_outcome = np.zeros(n_matches, dtype=int)
    opponent_outcome = np.zeros(n_matches, dtype=int) 

    for round_number in range(n_rounds):
        if round_number < memory:
            # Opening moves: the same bit for every row
            player_intended = players[:, round_number]
            opponent_intended = opponents[:, round_number]
        else:
            # For each row k, take the bit at position memory + outcome[k]
            player_intended = players[np.arange(n_matches) , memory + player_outcome]
            opponent_intended = opponents[np.arange(n_matches) , memory + opponent_outcome]

        # Noise: each intended action is flipped with probability p_noise
        player_mistakes = rng.random(n_matches) < p_noise
        opponent_mistakes = rng.random(n_matches) < p_noise
        player_action = np.where(player_mistakes, 1 - player_intended, player_intended)
        opponent_action = np.where(opponent_mistakes, 1 - opponent_intended, opponent_intended)

        player_scores += payoff_table[player_action, opponent_action]
        opponent_scores += payoff_table[opponent_action, player_action]
        player_C += player_action == 0
        opponent_C += opponent_action == 0

        # update new outcomes
        player_outcome = (4 * player_outcome + 2 * player_action + opponent_action) % n_outcomes
        opponent_outcome = (4 * opponent_outcome + 2 * opponent_action + player_action) % n_outcomes

    return player_scores, opponent_scores, player_C, opponent_C

