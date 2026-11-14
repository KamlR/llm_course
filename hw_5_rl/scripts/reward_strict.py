"""Строгая награда для RL-1 (GRPO): полное совпадение вызовов -> 1, иначе 0.

В TRL: reward_funcs=strict_reward; reference и tools берутся из датасета.
Разбор и проверка завершения общие для обеих наград, без запуска модели/GPU.
"""
from collections import Counter
from functools import lru_cache
import json
import re

from transformers import AutoConfig, AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
CALL = re.compile(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', re.S)
PARAM = re.compile(r'<parameter=([^>\n]+)>(.*?)</parameter>', re.S)
MARKUP = re.compile(r'</?(?:tool|function|parameter)')


@lru_cache(maxsize=1)
def runtime():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION).get_text_config()
    eos = config.eos_token_id or []
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    return tokenizer, Qwen3CoderToolParser(tokenizer), eos


def freeze(value):
    """Порядок ключей не важен; порядок списков важен. True != 1, число 1 == 1.0."""
    if isinstance(value, dict):
        return ('object', tuple(sorted((k, freeze(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ('array', tuple(map(freeze, value)))
    kind = 'number' if type(value) in (int, float) else type(value).__name__
    return (kind, value)


def parse_completion(token_ids, tools):
    """None = пустой/обрезанный/некорректный ответ; [] = текст без вызовов."""
    tokenizer, parser, eos = runtime()
    ids = list(token_ids)
    while ids and ids[-1] == tokenizer.pad_token_id and ids[-1] not in eos:
        ids.pop()
    if not ids or ids[-1] not in eos:
        return None
    # TRL может удалить спецтокены из completions: разбираем исходные token IDs.
    text = tokenizer.decode(ids[:-1], skip_special_tokens=False).strip()
    if re.search(r'<\|[^>]*\|>', text):
        return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if not text or '<think' in text or '</think' in text:
        return None
    blocks = list(CALL.finditer(text))
    if MARKUP.search(CALL.sub('', text)):
        return None
    if not blocks:
        if text.startswith(('[', '{', '```')):
            return None
        return [] if re.search(r'\w', text) else None
    for block in blocks:
        body = block[2]
        parameters = list(PARAM.finditer(body))
        names = [p[1] for p in parameters]
        if PARAM.sub('', body).strip() or len(names) != len(set(names)):
            return None
        if any(MARKUP.search(p[2]) for p in parameters):
            return None
    try:
        request = ChatCompletionRequest.model_validate({'model': MODEL, 'messages': [], 'tools': tools})
        parser.tools = request.tools or []
        parsed = parser.extract_tool_calls(text, request).tool_calls
        if len(parsed) != len(blocks):
            return None
        calls = [{'name': call.function.name, 'arguments': json.loads(call.function.arguments)} for call in parsed]
        json.dumps(calls, allow_nan=False)
        return calls
    except (ValueError, TypeError, KeyError):
        return None


def score_batch(completion_ids, reference, tools, score):
    rewards = []
    for ids, target, available in zip(completion_ids, reference, tools, strict=True):
        calls = parse_completion(ids, available)
        rewards.append(0.0 if calls is None else float(score(calls, target['calls'])))
    return rewards


def exact_match(predicted, expected):
    def bag(calls):
        return Counter((c['name'], freeze(c['arguments'])) for c in calls)
    return bag(predicted) == bag(expected)


def strict_reward(completions, completion_ids, reference, tools, **kwargs):
    """Интерфейс TRL: возвращает по одной награде 0/1 на сгенерированный ответ."""
    if len(completions) != len(completion_ids):
        raise ValueError('completions и completion_ids имеют разную длину')
    return score_batch(completion_ids, reference, tools, exact_match)
