#!/usr/bin/env python3
"""python3 scripts/prepare_data.py -> data/train.jsonl, data/val.jsonl.

Берём первый user/assistant каждого диалога, только JSON-описания инструментов.
Эталоны вызовов разбираем без исполнения кода; пустые/неподдерживаемые пропускаем.
90/10, seed=42; одинаковые первые запросы всегда в одном сплите.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from jsonschema import Draft7Validator, SchemaError, ValidationError
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
MARKERS = ['Here is a list of functions in JSON format that you can invoke:',
           '以下是可以调用的工具的JSON格式列表:']


def normalize_schema(schema):
    """Типы ToolACE -> JSON Schema; имена параметров и значения не меняем."""
    result = dict(schema)
    aliases = {'dict': 'object', 'int': 'integer', 'float': 'number', 'bool': 'boolean'}
    kind = result.get('type')
    if isinstance(kind, str):
        result['type'] = aliases.get(kind, kind)
    elif isinstance(kind, list):
        result['type'] = [aliases.get(t, t) for t in kind]
    if 'properties' in result:
        result['properties'] = {k: normalize_schema(v) for k, v in result['properties'].items()}
    for key in ['items', 'additionalProperties']:
        if isinstance(result.get(key), dict):
            result[key] = normalize_schema(result[key])
        elif isinstance(result.get(key), list):
            result[key] = [normalize_schema(v) for v in result[key]]
    for key in ['anyOf', 'allOf', 'oneOf']:
        if key in result:
            result[key] = [normalize_schema(v) for v in result[key]]
    return result


def parse_calls(text, names):
    """[имя функции(args), ...] -> calls. ast.literal_eval не исполняет код."""
    if not (text.startswith('[') and text.endswith(']')):
        raise ValueError('unsupported_response')
    body, calls = text[1:-1].strip(), []
    while body:
        name = next((n for n in sorted(names, key=len, reverse=True)
                     if body.startswith(n) and body[len(n):].lstrip().startswith('(')), None)
        if name is None:
            raise ValueError('unknown_call')
        rest = body[len(name):].lstrip()
        stack, quote, escaped, end = [], None, False, None
        for i, char in enumerate(rest):
            if quote:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ['"', "'"]:
                quote = char
            elif char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop() != {')': '(', ']': '[', '}': '{'}[char]:
                    raise ValueError('unbalanced_call')
                if not stack:
                    end = i + 1
                    break
        if end is None:
            raise ValueError('unbalanced_call')
        node = ast.parse('f' + rest[:end], mode='eval').body
        if not isinstance(node, ast.Call) or node.args or any(k.arg is None for k in node.keywords):
            raise ValueError('unsupported_arguments')
        # JSON true/false/null могут встречаться внутри списков и словарей.
        for child in ast.walk(node):
            for field, value in ast.iter_fields(child):
                if isinstance(value, ast.Name) and value.id in {'true', 'false', 'null'}:
                    setattr(child, field, ast.Constant({'true': True, 'false': False, 'null': None}[value.id]))
                elif isinstance(value, list):
                    setattr(child, field, [ast.Constant({'true': True, 'false': False, 'null': None}[v.id])
                            if isinstance(v, ast.Name) and v.id in {'true', 'false', 'null'} else v for v in value])
        args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        if len(args) != len(node.keywords):
            raise ValueError('duplicate_argument')
        calls.append({'name': names[name], 'arguments': args})
        body = rest[end:].strip()
        if body:
            if not body.startswith(',') or not body[1:].strip():
                raise ValueError('unsupported_response')
            body = body[1:].strip()
    if not calls:
        raise ValueError('empty_call_list')
    return calls


def main():
    rows = json.loads((ROOT / 'data/raw/toolace.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    counts, unique, seen = Counter(), [], set()
    for index, row in enumerate(rows):
        marker = next((m for m in MARKERS if m in row['system']), None)
        if marker is None:
            counts['skip_non_json_tools'] += 1
            continue
        prefix, tail = row['system'].split(marker, 1)
        source_tools, _ = json.JSONDecoder().raw_decode(tail.lstrip())
        user, answer = row['conversations'][:2]
        assert (user['from'], answer['from']) == ('user', 'assistant')
        text = answer['value'].strip()
        if not text:
            counts['skip_empty_answer'] += 1
            continue
        try:
            names, tools, validators = {}, [], {}
            for i, tool in enumerate(source_tools):
                original = tool['name']
                if not original or original in names:
                    raise ValueError('duplicate_or_empty_name')
                name = re.sub(r'[^a-zA-Z0-9_-]', '_', original).strip('_')[:55] or 'tool'
                if name in names.values():
                    name += f'_{i}'
                names[original] = name
                schema = normalize_schema(tool['parameters'])
                Draft7Validator.check_schema(schema)
                validators[name] = Draft7Validator(schema)
                tools.append({'type': 'function', 'function': {
                    'name': name, 'description': tool.get('description', ''), 'parameters': schema}})
            if text.startswith('['):
                calls = parse_calls(text, names)
            else:
                if text.startswith(('{', '<', '```')) or '=>' in text or any(n + '(' in text for n in names):
                    raise ValueError('unsupported_response')
                calls = []
            for call in calls:
                validator = validators[call['name']]
                validator.validate(call['arguments'])
                if set(call['arguments']) - set(validator.schema.get('properties', {})):
                    raise ValueError('unknown_argument')
            json.dumps(calls, allow_nan=False)  # Только значения, представимые в JSON.
        except (ValueError, TypeError, KeyError, SyntaxError, SchemaError, ValidationError):
            counts['skip_invalid_schema_or_reference'] += 1
            continue
        # Сохраняем даты и контекст из preamble, убираем старый формат вызовов.
        prefix = prefix.replace('请用<cot></cot>标识你的思考过程。', '').strip()
        prompt = [{'role': 'system', 'content': prefix}, {'role': 'user', 'content': user['value']}]
        rendered = tokenizer.apply_chat_template(prompt, tools=tools, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        target = {'role': 'assistant', 'content': '' if calls else text}
        if calls:
            target['tool_calls'] = [{'type': 'function', 'function': c} for c in calls]
        full = tokenizer.apply_chat_template(prompt + [target], tools=tools, tokenize=False, enable_thinking=False)
        assert full.startswith(rendered)
        if (len(tokenizer.encode(rendered, add_special_tokens=False)) > 3072 or
                len(tokenizer.encode(full[len(rendered):], add_special_tokens=False)) > 1024):
            counts['skip_too_long'] += 1
            continue
        key = json.dumps([prompt, tools], sort_keys=True, ensure_ascii=False)
        if key in seen:
            counts['skip_duplicate_prompt'] += 1
            continue
        seen.add(key)
        unique.append({'dialogue_id': f'toolace_{index:05d}', 'prompt': prompt,
                       'tools': tools, 'reference': {'calls': calls}})
    # Группируем даже совпадающие запросы с различными наборами инструментов.
    groups = defaultdict(list)
    for example in unique:
        query = ' '.join(example['prompt'][-1]['content'].casefold().split())
        groups[hashlib.sha256(query.encode()).hexdigest()].append(example)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    cut = int(len(keys) * .9)
    for split, selected in [('train', keys[:cut]), ('val', keys[cut:])]:
        examples = [example for key in selected for example in groups[key]]
        with (ROOT / f'data/{split}.jsonl').open('w') as handle:
            for example in examples:
                line = json.dumps(example, ensure_ascii=False, allow_nan=False)
                # Публикуем данные без похожих на GitHub-токены строк из ToolACE.
                line = re.sub(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})',
                              'REDACTED_GITHUB_TOKEN', line)
                handle.write(line + '\n')
        counts[split] = len(examples)
        counts[split + '_no_call'] = sum(not e['reference']['calls'] for e in examples)
    print(json.dumps({'seed': SEED, **counts}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
