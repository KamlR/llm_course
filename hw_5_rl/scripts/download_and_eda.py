#!/usr/bin/env python3
"""Скачать ToolACE и сделать базовый EDA на CPU (без весов модели).

В контейнере: python3 scripts/download_and_eda.py
Зависимости: из Dockerfile-3 + matplotlib==3.10.7 (уже установлен).
Выход: data/raw/toolace.json, results/eda/{summary.json,overview.png}.
Данные не фильтруются, train/val не создаются.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from collections import Counter

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MPLCONFIGDIR', '/cache/matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'Team-ACE/ToolACE'
DATA_REV = '6bda777c88d21e5a204703c1ee45597a8fa4f734'
MODEL = 'Qwen/Qwen3.5-2B'
MODEL_REV = '15852e8c16360a2fea060d615a32b45270f8a8fc'


def count_tools(system):
    """Количество описаний инструментов; схемы аргументов не преобразуем."""
    for marker in ['Here is a list of functions in JSON format that you can invoke:',
                   '以下是可以调用的工具的JSON格式列表:']:
        if marker in system:
            tools, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].lstrip())
            return len(tools), 'json'
    marker = 'Here are the tools you can use:'
    if marker not in system:
        if any(tag in system for tag in ['tool_name', '"name"', '<table']):
            raise ValueError('Незнакомый формат инструментов')
        return 0, 'no_section'
    block = system.split(marker, 1)[1].split('Please use the following format', 1)[0]
    patterns = [r'"tool_name"\s*:', r'<tool_name>', r'<tr>\s*<td>',
                r'(?m)^\s*-\s*\*\*tool_name\*\*:', r'(?m)^tool_name:']
    count = sum(len(re.findall(pattern, block)) for pattern in patterns)
    for table in re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', block, re.S):
        count += sum(name.strip() != 'tool_name'
                     for name in re.findall(r'(?m)^([^\n&]+)\s*&', table))
    if not count and block.strip():
        raise ValueError('Не удалось посчитать инструменты')
    return count, 'nonstandard'


def token_lengths(tokenizer, texts):
    lengths = []
    for start in range(0, len(texts), 128):
        batch = tokenizer(texts[start:start + 128], add_special_tokens=False,
                          truncation=False, return_attention_mask=False)
        lengths.extend(map(len, batch['input_ids']))
    return np.array(lengths)


def stats(values):
    return {'min': int(np.min(values)), 'median': float(np.median(values)),
            'mean': round(float(np.mean(values)), 2),
            'p95': round(float(np.percentile(values, 95)), 2), 'max': int(np.max(values))}


def fingerprint(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    raw = ROOT / 'data/raw/toolace.json'
    output = ROOT / 'results/eda'
    raw.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    print('Скачиваем ToolACE (при повторном запуске используется кеш)...', flush=True)
    cached = hf_hub_download(DATASET, 'data.json', repo_type='dataset', revision=DATA_REV)
    shutil.copyfile(cached, raw)
    rows = json.loads(raw.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    tokenizer.model_max_length = 10**9  # Считаем полные длины без усечения.

    prompts, answers, messages_per_dialogue, tools_per_dialogue = [], [], [], []
    roles, tool_formats, answer_types = Counter(), Counter(), Counter()
    empty_messages = duplicate_dialogues = duplicate_conversations = 0
    seen_dialogues, seen_conversations = set(), set()
    for row in rows:
        messages = row['conversations']
        for value, seen, kind in [(row, seen_dialogues, 'dialogue'),
                                  (messages, seen_conversations, 'conversation')]:
            key = fingerprint(value)
            if key in seen:
                duplicate_dialogues += kind == 'dialogue'
                duplicate_conversations += kind == 'conversation'
            seen.add(key)
        count, tool_format = count_tools(row['system'])
        tools_per_dialogue.append(count)
        tool_formats[tool_format] += 1
        messages_per_dialogue.append(len(messages))
        history = [{'role': 'system', 'content': row['system']}]
        for message in messages:
            role, text = message['from'], message['value']
            roles[role] += 1
            empty_messages += not text.strip()
            if role == 'assistant':
                prompts.append(tokenizer.apply_chat_template(
                    history, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                answers.append(text)
                # Это только внешний вид ответа, а не проверка вызовов функций.
                if not text.strip():
                    kind = 'empty'
                elif text.lstrip().startswith(('[', '{', '<', '```')) or '=>' in text:
                    kind = 'structured_like'
                elif history[-1]['role'] == 'tool':
                    kind = 'text_after_tool'
                else:
                    kind = 'text_after_user'
                answer_types[kind] += 1
            history.append({'role': role, 'content': text})
    print(f'Диалогов: {len(rows)}; ответов: {len(answers)}. Считаем токены...', flush=True)
    prompt_tokens = token_lengths(tokenizer, prompts)
    response_tokens = token_lengths(tokenizer, answers)
    summary = {
        'dataset': DATASET, 'dataset_revision': DATA_REV,
        'data_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'tokenizer': MODEL, 'tokenizer_revision': MODEL_REV,
        'dialogues': len(rows), 'messages_by_role': dict(roles),
        'messages_per_dialogue': stats(messages_per_dialogue),
        'available_tools_per_dialogue': stats(tools_per_dialogue),
        'tool_description_formats': dict(tool_formats),
        'prompt_tokens': stats(prompt_tokens), 'response_tokens': stats(response_tokens),
        'response_appearance_heuristic': dict(answer_types),
        'empty_messages': empty_messages, 'duplicate_dialogues_beyond_first': duplicate_dialogues,
        'duplicate_conversations_without_system_beyond_first': duplicate_conversations,
        'prompt_fits_4096_with_1024_reserved_percent': round(float(np.mean(prompt_tokens + 1024 <= 4096) * 100), 2),
        'notes': [
            'Prompt: исходный system с инструментами + история до ответа + generation prompt Qwen без thinking.',
            'Response: исходный текст ответа без служебных токенов. После RL-конвертации длины пересчитать.',
            'structured_like — только эвристика по разметке, не число корректных вызовов; text_after_user не равен Irrelevance.',
            'Дубликаты без system могут иметь разные инструменты. Ничего не удалено из датасета.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, values, title, xlabel in [
        (axes[0, 0], prompt_tokens, 'Длина промпта: исходный формат', 'Токены Qwen'),
        (axes[0, 1], response_tokens, 'Длина текста ответа', 'Токены Qwen'),
    ]:
        ax.hist(values, bins=45, color='#2866A5', edgecolor='white', linewidth=.3)
        ax.axvline(np.percentile(values, 95), color='#D56A26', ls='--', label='95-й процентиль')
        ax.set(title=title, xlabel=xlabel, ylabel='Ответы ассистента')
        ax.legend()
    for ax, values, title in [
        (axes[1, 0], messages_per_dialogue, 'Сообщения в диалоге (без system)'),
        (axes[1, 1], tools_per_dialogue, 'Доступные инструменты в диалоге'),
    ]:
        counts = Counter(values)
        ax.bar(sorted(counts), [counts[x] for x in sorted(counts)], color='#168277')
        ax.set(title=title, xlabel='Количество', ylabel='Диалоги')
        ax.set_xticks(sorted(counts))
    fig.suptitle(f'ToolACE: {len(rows):,} диалогов, {len(answers):,} ответов ассистента', fontsize=14)
    fig.savefig(output / 'overview.png', dpi=160)
    plt.close(fig)
    print(f'Готово: {output}/summary.json и overview.png', flush=True)


if __name__ == '__main__':
    main()
