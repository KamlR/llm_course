#!/usr/bin/env python3
"""Оценка base, Full FT и LoRA на официальном тесте BANKING77.

Запуск после завершения обучения обеих моделей, из папки проекта:
    /home/edu/.venv/bin/python evaluate_models.py --check-only
    /home/edu/.venv/bin/python evaluate_models.py

По умолчанию: физический GPU 1, результаты в outputs/evaluation.
metrics.md содержит отдельную таблицу для Accuracy и Macro-F1.
manual_review.md содержит по пять одинаковых примеров для каждой модели.
Ответы и автоматическая проверка заполняются скриптом; ручные комментарии
нужно добавить самостоятельно после просмотра ответов.
Все веса и данные читаются локально, скачивание не требуется.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--full-ft-dir", type=Path, default=Path("outputs/full_ft"))
    parser.add_argument("--lora-dir", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--gpu", default="1", help="Физический индекс GPU")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--manual-examples", type=int, default=5, choices=range(5, 11))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="Проверить данные, пути и токенизацию без загрузки весов")
    args = parser.parse_args()
    if not args.gpu.isdigit() or args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("Укажите индекс одного GPU и положительные batch-size/max-new-tokens")
    return args


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(targets, predictions, label_names):
    """Macro-F1 по всем классам. Невалидный ответ — ошибка, а не пропуск."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("Нужны непустые списки одинаковой длины")
    labels = set(label_names)
    if not set(targets) <= labels:
        raise ValueError("Неизвестная эталонная категория")
    true_counts = Counter(targets)
    predicted_counts = Counter(predictions)
    correct_counts = Counter(t for t, p in zip(targets, predictions) if t == p)
    f1 = []
    for label in label_names:
        # 2*TP + FP + FN = число истинных + число предсказанных объектов класса.
        denominator = true_counts[label] + predicted_counts[label]
        f1.append(2 * correct_counts[label] / denominator if denominator else 0.0)
    return {
        "accuracy": sum(correct_counts.values()) / len(targets),
        "macro_f1": sum(f1) / len(label_names),
        "test_examples": len(targets),
        "invalid_predictions": sum(p not in labels for p in predictions),
    }


def select_manual_ids(targets, count, seed):
    """Выбор до генерации: по одному обращению из случайных разных классов."""
    rng = random.Random(seed)
    classes = rng.sample(sorted(set(targets)), count)
    return [rng.choice([i for i, target in enumerate(targets) if target == label]) for label in classes]


def prepare_data(root, full_ft_dir, lora_dir, base_path):
    from datasets import ClassLabel, load_dataset
    from transformers import AutoTokenizer

    manifest = read_json(root / "data/banking77_split.json")
    labels = manifest["label_names"]
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Ожидаются 77 уникальных категорий BANKING77")
    prompt = (
        "Classify the intent of the following banking customer message. "
        "Choose exactly one category from the list below. "
        "Reply with the category name exactly as written, without explanations.\n\n"
        "Allowed categories:\n" + "\n".join(labels)
    )
    for directory in (full_ft_dir, lora_dir):
        if read_json(directory / "data_split.json") != manifest:
            raise ValueError(f"Отличается разбиение данных: {directory}")
        if read_json(directory / "label_names.json") != labels:
            raise ValueError(f"Отличаются категории: {directory}")
        if (directory / "system_prompt.txt").read_text(encoding="utf-8") != prompt:
            raise ValueError(f"Отличается инструкция: {directory}")

    dataset_dir = root / "data/banking77"
    test_files = sorted((dataset_dir / "data").glob("test-*.parquet"))
    expected_files = sorted(k for k in manifest["source_files_sha256"] if Path(k).name.startswith("test-"))
    if not test_files or [str(p.relative_to(dataset_dir)) for p in test_files] != expected_files:
        raise ValueError("Набор тестовых файлов отличается от сохранённого")
    for path in test_files:
        relative = str(path.relative_to(dataset_dir))
        if sha256_file(path) != manifest["source_files_sha256"][relative]:
            raise ValueError(f"Изменился тестовый файл {path}")
    dataset = load_dataset(
        "parquet", data_files={"test": [str(p) for p in test_files]}, split="test",
        cache_dir=str(dataset_dir / ".dataset_cache"),
    )
    if len(dataset) != 3080 or not isinstance(dataset.features["label"], ClassLabel):
        raise ValueError("Ожидается официальный test BANKING77: 3080 примеров с ClassLabel")
    if dataset.features["label"].names != labels:
        raise ValueError("Категории теста отличаются от обучающих")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    # Один и тот же токенизатор и шаблон для всех моделей; проверяем сохранённые копии.
    for directory in (full_ft_dir / "model", lora_dir / "adapter"):
        saved = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        if (saved.get_vocab() != tokenizer.get_vocab()
                or saved.chat_template != tokenizer.chat_template
                or saved.special_tokens_map != tokenizer.special_tokens_map):
            raise ValueError(f"Изменён токенизатор: {directory}")
    tokenizer.padding_side = "left"  # Генерация decoder-only модели в батче.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for i, row in enumerate(dataset):
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Пустое обращение test[{i}]")
        rows.append({"test_row_id": i, "text": row["text"], "target": labels[row["label"]]})
    encoded = [tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": row["text"]}],
        tokenize=True, add_generation_prompt=True,
    ) for row in rows]
    return rows, encoded, labels, tokenizer, manifest


def generate_predictions(model, tokenizer, encoded, batch_size, max_new_tokens):
    import torch
    from tqdm.auto import tqdm

    outputs = []
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(encoded), batch_size), desc="Генерация", unit="batch"):
            sequences = encoded[start:start + batch_size]
            inputs = tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in sequences],
                padding=True, return_tensors="pt",
            ).to(model.device)
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Удаляем весь вход, включая левый padding, оставляем только ответ.
            answers = generated[:, inputs["input_ids"].shape[1]:].cpu().tolist()
            for ids in answers:
                outputs.append({
                    "response": tokenizer.decode(ids, skip_special_tokens=True),
                    "stopped_by_eos": any(token in eos_ids for token in ids),
                })
    return outputs


def save_reports(output_dir, metrics, predictions, manual_ids):
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    write_json(output_dir / "metrics.json", metrics)
    lines = ["# Оценка на тесте BANKING77", "",
             "Все модели оценены на одних и тех же 3080 обращениях. Генерация greedy.", "",
             "Ответ должен совпасть с названием категории; удаляются только пробелы и переносы по краям. "
             "Объяснения вместо категории и неизвестные категории считаются ошибками.", ""]
    for key, title, explanation in [
        ("accuracy", "Accuracy", "Доля обращений с правильно предсказанной категорией."),
        ("macro_f1", "Macro-F1", "Среднее F1 по всем 77 категориям: каждый класс имеет одинаковый вес."),
    ]:
        lines.extend([f"## {title}", "", explanation, "", f"| Модель | {title}, % |", "|---|---:|"])
        lines.extend(f"| {row['model']} | {100 * row[key]:.2f} |" for row in metrics)
        lines.append("")
    lines.extend(["В CSV/JSON метрики записаны в диапазоне от 0 до 1.", ""])
    report = "\n".join(lines)
    (output_dir / "metrics.md").write_text(report, encoding="utf-8")

    manual = ["# Примеры для ручной оценки", "",
              "Одинаковые обращения для всех моделей выбраны до генерации: по одному из случайных разных классов. "
              "Автоматическое совпадение категории не заменяет ручной разбор; комментарии нужно заполнить после просмотра.", ""]
    for model_name, records in predictions.items():
        manual.extend([f"## {model_name}", ""])
        for number, row_id in enumerate(manual_ids, 1):
            row = records[row_id]
            # Отступы защищают Markdown-разметку от символов в исходном тексте/ответе.
            manual.extend([f"### Пример {number} — test[{row_id}]", "", "Обращение:", ""])
            manual.extend("    " + line for line in row["text"].splitlines())
            manual.extend(["", f"Правильная категория: `{row['target']}`", "", "Ответ модели:", ""])
            manual.extend("    " + line for line in (row["response"] or "[пустой ответ]").splitlines())
            manual.extend(["", f"Совпадение категории: **{'да' if row['correct'] else 'нет'}**.", "",
                           "Ручной комментарий: _заполнить после просмотра_.", ""])
    (output_dir / "manual_review.md").write_text("\n".join(manual), encoding="utf-8")
    return report


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Запускайте обычным python на одном GPU, без torchrun")
    import importlib.metadata
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed

    root = args.project_dir.expanduser().resolve()
    full_ft_dir, lora_dir, output_dir = [root / p.expanduser() for p in (args.full_ft_dir, args.lora_dir, args.output_dir)]
    base_path = root / "models/Qwen2.5-1.5B-Instruct"
    for directory, config in [(base_path, "config.json"), (full_ft_dir / "model", "config.json"),
                              (lora_dir / "adapter", "adapter_config.json")]:
        if not (directory / config).is_file() or not list(directory.glob("*.safetensors")):
            raise FileNotFoundError(f"Нет сохранённой модели в {directory}. Дождитесь завершения обучения.")
    if not args.check_only and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} уже содержит результаты; задайте другой --output-dir")
    rows, encoded, labels, tokenizer, manifest = prepare_data(root, full_ft_dir, lora_dir, base_path)
    longest_label = max(len(tokenizer.encode(label, add_special_tokens=False)) for label in labels)
    if args.max_new_tokens < longest_label + 1:
        raise ValueError("max-new-tokens недостаточно для самой длинной категории и EOS")
    for directory in (base_path, full_ft_dir / "model"):
        context = read_json(directory / "config.json")["max_position_embeddings"]
        if max(map(len, encoded)) + args.max_new_tokens > context:
            raise ValueError(f"Вход и ответ не помещаются в контекст {directory}")
    targets = [row["target"] for row in rows]
    manual_ids = select_manual_ids(targets, args.manual_examples, args.seed)
    print(f"Test: {len(rows)}, классов: {len(labels)}, примеры для разбора: {manual_ids}", flush=True)
    if args.check_only:
        print("Проверка пройдена. Веса не загружались, генерация не запускалась.")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Нужен один видимый CUDA GPU с поддержкой BF16")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_config.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_path": str(base_path), "full_ft_dir": str(full_ft_dir), "lora_dir": str(lora_dir),
        "test_files_sha256": {k: v for k, v in manifest["source_files_sha256"].items() if Path(k).name.startswith("test-")},
        "manual_test_row_ids": manual_ids, "generation": "greedy; do_sample=False; num_beams=1",
        "matching": "case-sensitive exact category match after str.strip(); invalid answers count as errors",
        "macro_f1": "All 77 classes, zero_division=0",
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "datasets")},
    })
    metrics, predictions = [], {}
    generation_config = GenerationConfig.from_pretrained(str(base_path), local_files_only=True)
    for name, slug, path in [("Base", "base", base_path), ("Full FT", "full_ft", full_ft_dir / "model"),
                             ("LoRA", "lora", base_path)]:
        set_seed(args.seed)
        print(f"\n{name}: загрузка модели", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda:0")
        if slug == "lora":
            model = PeftModel.from_pretrained(model, str(lora_dir / "adapter"), local_files_only=True, is_trainable=False)
        model.generation_config = deepcopy(generation_config)
        answers = generate_predictions(model, tokenizer, encoded, args.batch_size, args.max_new_tokens)
        records = []
        for row, answer in zip(rows, answers):
            predicted = answer["response"].strip()
            records.append({**row, **answer, "predicted_label": predicted,
                            "valid_label": predicted in labels, "correct": predicted == row["target"]})
        scores = classification_metrics(targets, [r["predicted_label"] for r in records], labels)
        metrics.append({"model": name, **scores, "without_eos": sum(not r["stopped_by_eos"] for r in records)})
        predictions[name] = records
        with (output_dir / f"predictions_{slug}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Промежуточные результаты сохраняются после каждой модели.
        save_reports(output_dir, metrics, predictions, manual_ids)
        print(json.dumps(metrics[-1], ensure_ascii=False, indent=2), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print("\n" + save_reports(output_dir, metrics, predictions, manual_ids))
    print(f"Результаты: {output_dir}\nРучной разбор: {output_dir / 'manual_review.md'}")


if __name__ == "__main__":
    main()
