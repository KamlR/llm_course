"""Частичная награда для RL-2 (GRPO) и RL-3 (Dr. GRPO), диапазон [0, 1].

Для пары вызовов: s=0 при разных именах, иначе s=0.25+0.75*F1 аргументов.
F1 = 2 * число совпавших пар (имя параметра, значение) / (|pred_args|+|gold_args|).
Для двух пустых словарей F1=1. Вложенное значение сравнивается целиком.
Итог R = 2 * сумма s при оптимальном сопоставлении один-к-одному / (|pred|+|gold|).
Без ожидаемых вызовов: 1 только за корректный непустой ответ без вызовов.
Пустые, обрезанные и некорректно размеченные ответы: 0, как в строгой награде.
"""
from scipy.optimize import linear_sum_assignment

if __package__:
    from .reward_strict import freeze, score_batch
else:
    from reward_strict import freeze, score_batch


def call_similarity(predicted, expected):
    if predicted['name'] != expected['name']:
        return 0.0
    p, g = predicted['arguments'], expected['arguments']
    matched = sum(k in g and freeze(v) == freeze(g[k]) for k, v in p.items())
    f1 = 2 * matched / (len(p) + len(g)) if p or g else 1.0
    return 0.25 + 0.75 * f1


def partial_match(predicted, expected):
    if not expected:
        return float(not predicted)
    if not predicted:
        return 0.0
    weights = [[call_similarity(p, g) for g in expected] for p in predicted]
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return 2 * sum(weights[i][j] for i, j in zip(rows, cols)) / (len(predicted) + len(expected))


def partial_reward(completions, completion_ids, reference, tools, **kwargs):
    """В TRL: reward_funcs=partial_reward. Максимум 1 только при полном совпадении."""
    if len(completions) != len(completion_ids):
        raise ValueError('completions и completion_ids имеют разную длину')
    return score_batch(completion_ids, reference, tools, partial_match)
