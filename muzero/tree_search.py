import math
from typing import List

import numpy
from torch import Tensor
from torch.functional import F

from muzero.context import MinMaxStats, MuZeroContext
from muzero.game import Node
from muzero.networks import MuZeroNetwork


def run_mcts(
    context: MuZeroContext,
    root: Node,
    action_history: List[int],
    network: MuZeroNetwork,
):
    min_max_stats = MinMaxStats(context.known_bounds)

    for _ in range(context.num_simulations):
        last_action = action_history[-1] if action_history else None
        node = root
        search_path = [node]

        while node.expanded():
            last_action, node = pick_child(context, node, min_max_stats)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        hidden_state, reward_logits, policy_logits, value_logits = (
            network.recurrent_inference(parent.hidden_state, Tensor([[last_action]]))
        )

        reward = network.support_to_scalar(reward_logits).item()
        value = network.support_to_scalar(value_logits).item()

        expand_node(node, hidden_state, policy_logits, reward)

        backpropagate(
            search_path,
            value,
            context.discount,
            min_max_stats,
        )


def pick_child(context: MuZeroContext, node: Node, min_max_stats: MinMaxStats):
    ucb_scores = [
        (ucb_score(context, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    ]

    max_ucb_score, action, _ = max(ucb_scores)

    # If two actions have the same score, we break the tie randomly.
    action = numpy.random.choice(
        [action for ucb_score, action, _ in ucb_scores if ucb_score == max_ucb_score]
    )

    return action, node.children[action]


def ucb_score(
    context: MuZeroContext, parent: Node, child: Node, min_max_stats: MinMaxStats
) -> float:
    """
    Upper confidence bound for trees. It trades off exploration and exploitation.

    The score for a node is based on its value, plus an exploration bonus based on
    the prior.
    """
    pb_c = (
        math.log((parent.visit_count + context.pb_c_base + 1) / context.pb_c_base)
        + context.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.probability
    value_score = (
        child.reward + context.discount * min_max_stats.normalize(child.value())
        if child.visit_count > 0
        else 0
    )
    return prior_score + value_score


def expand_node(
    node: Node,
    hidden_state: Tensor,
    policy_logits: Tensor,
    reward: Tensor,
):
    node.hidden_state = hidden_state
    node.reward = reward
    policy = F.softmax(policy_logits, dim=1)
    node.children = {action: Node(p) for action, p in enumerate(policy.squeeze())}


def add_exploration_noise(
    root_dirichlet_alpha: float,
    root_exploration_fraction: float,
    node: Node,
):
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([root_dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
        node.children[a].probability = (
            node.children[a].probability * (1 - root_exploration_fraction)
            + n * root_exploration_fraction
        )


def backpropagate(
    search_path: List[Node],
    value: float,
    discount: float,
    min_max_stats: MinMaxStats,
):
    for node in reversed(search_path):
        node.values_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value
