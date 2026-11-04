from dataclasses import dataclass

import numpy as np

from ipd.strategies import Strategy, first_action, next_action

C: int = 0 # Cooperate
D: int = 1 # Defect


def flip(action: int) -> int:
    """Return the opposite action: C -> D; D -> C"""
    return 1 - action

@dataclass
class PayoffMatrix:
    
    T: float = 5.0 # Temptation (I defect, the opponent cooperates)
    R: float = 3.0 # Reward (We both cooperate)
    P: float = 1.0 # Punishment (We both defect)
    S: float = 0.0 # Sucker (I cooperate, the opponent defects)
    
    def __post_init__(self) -> None: # set the rules that our payoff model must satisfy:
        # T > R > P > S (tempation provides me highest score)
        if not (self.T > self.R > self.P > self.S):
            raise ValueError("Payoffs must satisfy T > R > P > S")
        
        # 2R > T + S (cooperation provides better result than alternating C/D)
        if not (2* self.R > self.T + self.S):
            raise ValueError("Payoffs must satisfy 2R > T + S")
        
    def scores(self, a:int, b:int) -> tuple[float, float]:
        if a == C and b == C: 
            return self.R, self.R # both cooperate, get reward
        if a == C and b == D:
            return self.S, self.T 
        if a == D and b == C: 
            return self.T, self.S
        return self.P, self.P  # both defect, get punishment


def apply_noise(action: int, p_noise: float, rng: np.random.Generator) -> int:
    """With probability p_noise the intended action is flipped (a mistake)."""
    if rng.random() < p_noise:
        return flip(action)
    return action


def play_match(
    player: Strategy,
    opponent: Strategy,
    n_rounds: int = 150,
    p_noise: float = 0.0,
    payoffs: PayoffMatrix | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Play an iterated match and return the total scores (player, opponent)."""
    if payoffs is None:
        payoffs = PayoffMatrix()
    if rng is None:
        rng = np.random.default_rng()

    player_score = 0.0
    opponent_score = 0.0
    player_last = C  # placeholders, only read from round 2 on
    opponent_last = C

    for round_number in range(n_rounds):
        if round_number == 0:
            player_intended = first_action(player)
            opponent_intended = first_action(opponent)
        else:
            # Each player reads the previous round from its own side (own action first)
            player_intended = next_action(player, player_last, opponent_last)
            opponent_intended = next_action(opponent, opponent_last, player_last)

        player_action = apply_noise(player_intended, p_noise, rng)
        opponent_action = apply_noise(opponent_intended, p_noise, rng)

        player_payoff, opponent_payoff = payoffs.scores(player_action, opponent_action)
        player_score += player_payoff
        opponent_score += opponent_payoff

        # Strategies react to what was actually played, mistakes included
        player_last = player_action
        opponent_last = opponent_action

    return player_score, opponent_score

