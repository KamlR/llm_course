#!/usr/bin/env python3
"""LoRA-дообучение исходной Qwen2.5-1.5B-Instruct на BANKING77.

На сервере из /home/edu/lora_mavletova:
    /home/edu/.venv/bin/python -m pip install peft==0.17.1
    /home/edu/.venv/bin/python train_lora.py --check-only
    /home/edu/.venv/bin/python train_lora.py

Пример отдельного эксперимента с рангом:
    /home/edu/.venv/bin/python train_lora.py --r 8 --output-dir outputs/lora_r8

По умолчанию используется физический GPU 1. Модель читается из
models/Qwen2.5-1.5B-Instruct (исходные веса, не результат Full FT).
Датасет: data/banking77. Разбиение должно совпадать с
сохранённым Full FT файлом data/banking77_split.json.

Настройки LoRA: r=16, alpha=32, dropout=0.05, q/k/v/o-проекции attention.
Learning rate=2e-4; это отдельная настройка LoRA, у Full FT использовалось
2e-5. Остальные основные настройки обучения совпадают с Full FT.
В outputs/lora сохраняются адаптер, токенизатор, метрики, логи и график.
Для инференса понадобятся исходная модель и сохранённый адаптер.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--gpu", default="1", help="Физический индекс GPU в nvidia-smi")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-only", action="store_true", help="Проверить данные и маску loss без загрузки весов и обучения")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Имена слоёв через пробел; также поддерживается all-linear",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction должен быть между 0 и 1")
    for key in ["epochs", "learning_rate", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "max_length", "logging_steps"]:
        if getattr(args, key) <= 0:
            parser.error(f"{key} должен быть положительным")
    if not args.gpu.isdigit():
        parser.error("--gpu должен содержать индекс одной карты, например 1")
    if args.r <= 0 or args.alpha <= 0:
        parser.error("--r и --alpha должны быть положительными")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout должен быть в диапазоне [0, 1)")
    if "all-linear" in args.target_modules and args.target_modules != ["all-linear"]:
        parser.error("all-linear указывается отдельно от имён модулей")
    if len(set(args.target_modules)) != len(args.target_modules):
        parser.error("В --target-modules есть повторяющиеся имена")
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_system_prompt(label_names):
    return (
        "Classify the intent of the following banking customer message. "
        "Choose exactly one category from the list below. "
        "Reply with the category name exactly as written, without explanations.\n\n"
        "Allowed categories:\n" + "\n".join(label_names)
    )


def make_messages(text, system_prompt):
    """Этот же запрос нужно использовать для base, Full FT и LoRA при оценке."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def load_and_split(dataset_dir, validation_fraction, seed):
    from datasets import ClassLabel, load_dataset

    train_files = sorted((dataset_dir / "data").glob("train-*.parquet"))
    test_files = sorted((dataset_dir / "data").glob("test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Не найдены train/test Parquet в {dataset_dir / 'data'}")
    # Читаем только train. Test фиксируем контрольными суммами, но не используем.
    dataset = load_dataset(
        "parquet", data_files={"train": [str(p) for p in train_files]},
        split="train", cache_dir=str(dataset_dir / ".dataset_cache"),
    )
    if not isinstance(dataset.features["label"], ClassLabel):
        raise ValueError("Ожидается ClassLabel с исходным соответствием ID и названий")
    label_names = dataset.features["label"].names
    if len(label_names) != 77 or len(dataset) != 10003:
        raise ValueError("Ожидается исходный BANKING77: 10003 train-примера, 77 классов")
    if any(not isinstance(text, str) or not text.strip() for text in dataset["text"]):
        raise ValueError("В данных есть пустые или некорректные тексты")
    dataset = dataset.add_column("row_id", list(range(len(dataset))))
    split = dataset.train_test_split(
        test_size=validation_fraction, seed=seed, stratify_by_column="label",
    )
    train, validation = split["train"], split["test"]
    manifest = {
        "dataset": "PolyAI/banking77", "seed": seed,
        "validation_fraction": validation_fraction,
        "method": "datasets.train_test_split(stratify_by_column='label')",
        "source_files_sha256": {
            str(p.relative_to(dataset_dir)): sha256_file(p)
            for p in train_files + test_files
        },
        "label_names": label_names,
        "train_row_ids": train["row_id"],
        "validation_row_ids": validation["row_id"],
        "test": "Official test split, unchanged and not used for training or model selection",
    }
    return train, validation, label_names, manifest


def tokenize_dataset(dataset, tokenizer, label_names, system_prompt, max_length):
    """Маска -100 закрывает prompt; цель — только категория и EOS."""
    def encode(row):
        messages = make_messages(row["text"], system_prompt)
        target = label_names[row["label"]]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=True, add_generation_prompt=False,
        )
        # Проверяем границу prompt/answer, чтобы не ошибиться с маской.
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Граница ответа не совпала с prompt, row_id={row['row_id']}")
        answer_ids = full_ids[len(prompt_ids):]
        if tokenizer.eos_token_id not in answer_ids:
            raise ValueError("В ответе отсутствует EOS")
        # Конечный перенос строки шаблона после EOS не включаем в пример.
        answer_ids = answer_ids[:answer_ids.index(tokenizer.eos_token_id) + 1]
        if tokenizer.decode(answer_ids[:-1]) != target:
            raise ValueError("Токены целевого ответа не совпали с категорией")
        input_ids = prompt_ids + answer_ids
        if len(input_ids) > max_length:
            raise ValueError(
                f"row_id={row['row_id']}: {len(input_ids)} токенов > max_length={max_length}. "
                "Увеличьте --max-length: автоматическое обрезание отключено."
            )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    return dataset.map(encode, remove_columns=dataset.column_names, desc="Подготовка токенов и маски loss")


class CompletionCollator:
    """Дополняет входы PAD справа, а labels — значениями -100."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        import torch

        inputs = [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in examples]
        batch = self.tokenizer.pad(inputs, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        length = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [x["labels"] + [-100] * (length - len(x["labels"])) for x in examples],
            dtype=torch.long,
        )
        return batch



def save_loss_plot(log_history, output_dir):
    """Сохраняет график без графического интерфейса — подходит для SSH."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_logs = [row for row in log_history if "loss" in row]
    eval_logs = [row for row in log_history if "eval_loss" in row]
    fig, ax = plt.subplots(figsize=(9, 5))
    if train_logs:
        ax.plot(
            [row["step"] for row in train_logs],
            [row["loss"] for row in train_logs],
            label="Train loss",
        )
    if eval_logs:
        ax.plot(
            [row["step"] for row in eval_logs],
            [row["eval_loss"] for row in eval_logs],
            marker="o", label="Validation loss",
        )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("LoRA fine-tuning on BANKING77")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "loss_curve.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"График loss: {plot_path}", flush=True)



def make_lora_config(args):
    try:
        from peft import LoraConfig, TaskType
    except ModuleNotFoundError as error:
        if error.name != "peft":
            raise
        raise RuntimeError(
            "Установите PEFT в окружение запуска: "
            "/home/edu/.venv/bin/python -m pip install peft==0.17.1"
        ) from error
    target_modules = "all-linear" if args.target_modules == ["all-linear"] else args.target_modules
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
        use_rslora=False,  # Обычная LoRA: масштаб обновления alpha / r.
        use_dora=False,
    )


def add_lora_adapters(base_model, lora_config):
    from peft import get_peft_model

    # Проверяем каждое явно указанное имя: опечатка не должна тихо менять эксперимент.
    if isinstance(lora_config.target_modules, (list, set)):
        module_names = [name for name, _ in base_model.named_modules()]
        missing = [target for target in lora_config.target_modules if not any(
            name == target or name.endswith("." + target) for name in module_names
        )]
        if missing:
            raise ValueError(f"В модели нет target_modules: {missing}")
    model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=True)
    # PEFT замораживает исходные веса; обучаемыми остаются только матрицы LoRA.
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Ожидались только обучаемые LoRA-параметры")
    return model


def main():
    args = parse_args()
    # Должно выполняться до импорта torch и инициализации CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Запускайте обычным python на одном GPU, без torchrun")

    
    import importlib.metadata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    lora_config = make_lora_config(args)
    root = args.project_dir.expanduser().resolve()
    model_path = root / "models/Qwen2.5-1.5B-Instruct"
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    split_path = root / "data/banking77_split.json"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Не найдена модель: {model_path}")
    if not args.check_only and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} уже содержит результаты. Укажите другой --output-dir")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_raw, validation_raw, label_names, manifest = load_and_split(
        root / "data/banking77", args.validation_fraction, args.seed,
    )
    # Используем строго те же примеры, что и в завершённом Full FT.
    if not split_path.is_file():
        raise FileNotFoundError(f"Нужно разбиение от Full FT: {split_path}")
    if json.loads(split_path.read_text()) != manifest:
        raise ValueError(f"Разбиение отличается от {split_path}; используйте те же данные, seed и validation_fraction")
    system_prompt = make_system_prompt(label_names)
    full_ft_prompt = root / "outputs/full_ft/system_prompt.txt"
    if full_ft_prompt.exists() and full_ft_prompt.read_text() != system_prompt:
        raise ValueError("Инструкция отличается от использованной в Full FT")
    train = tokenize_dataset(train_raw, tokenizer, label_names, system_prompt, args.max_length)
    validation = tokenize_dataset(validation_raw, tokenizer, label_names, system_prompt, args.max_length)
    collator = CompletionCollator(tokenizer)
    batch = collator([train[i] for i in range(min(2, len(train)))])
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert torch.all((batch["labels"] != -100).sum(dim=1) > 0)
    del batch
    data_stats = {
        "train_examples": len(train), "validation_examples": len(validation), "classes": len(label_names),
        "max_train_tokens": max(len(x) for x in train["input_ids"]),
        "max_validation_tokens": max(len(x) for x in validation["input_ids"]),
    }
    print(json.dumps(data_stats, indent=2), flush=True)
    if args.check_only:
        print("Проверка пройдена: данные, токенизация и маска loss корректны. Веса не загружались.")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Требуется один видимый CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Выбранный GPU не поддерживает bfloat16")

    config = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="linear", warmup_ratio=0.1, weight_decay=0.01,
        optim="adamw_torch", bf16=True, fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        eval_strategy="epoch", prediction_loss_only=True, label_names=["labels"],
        logging_strategy="steps", logging_steps=args.logging_steps, logging_first_step=True,
        save_strategy="no", report_to="none",
        seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0,
        skip_memory_metrics=True,  # Иначе Trainer может сбросить наши пиковые счётчики CUDA.
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "data_split.json", manifest)
    write_json(output_dir / "label_names.json", label_names)
    (output_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "datasets", "accelerate", "peft"]}
    write_json(output_dir / "run_config.json", {
        "method": "lora", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "training_arguments": config.to_dict(), "versions": versions, "data": data_stats,
        "model_path": str(model_path), "physical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "loss": "Category tokens and EOS only; prompt and padding have label -100",
        "checkpoint": "Final epoch; no best-checkpoint selection",
        "lora": {
            "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
            "target_modules": args.target_modules, "scaling_alpha_over_r": args.alpha / args.r,
            "bias": "none", "use_rslora": False, "use_dora": False,
            "adapter_dtype": "float32 (PEFT autocast_adapter_dtype=True)",
        },
        "learning_rate_note": "LoRA uses 2e-4 by default; Full FT used 2e-5",
    })

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    base_params = sum(p.numel() for p in model.parameters())
    model = add_lora_adapters(model, lora_config)
    model.config.use_cache = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100.0 * trainable_params / total_params
    print(f"LoRA: {trainable_params:,} / {total_params:,} обучаемых параметров ({trainable_percent:.3f}%)", flush=True)
    trainer = Trainer(
        model=model, args=config, train_dataset=train, eval_dataset=validation,
        data_collator=collator, processing_class=tokenizer,
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    metrics = {
        "method": "lora", "total_parameters": total_params,
        "base_model_parameters": base_params,
        "trainable_parameters": trainable_params, "trainable_percent": trainable_percent,
        "lora_r": args.r, "lora_alpha": args.alpha, "lora_dropout": args.dropout,
        "target_modules": args.target_modules,
        "training_wall_seconds": elapsed,
        "timing_scope": "trainer.train including scheduled validation; excludes loading, tokenization and final saving",
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "memory_scope": "PyTorch allocations in this process during trainer.train, including model and optimizer",
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch, **result.metrics,
    }
    eval_logs = [row for row in trainer.state.log_history if "eval_loss" in row]
    if eval_logs:
        metrics["final_validation_loss"] = eval_logs[-1]["eval_loss"]
    write_json(output_dir / "training_metrics.json", metrics)
    write_json(output_dir / "log_history.json", trainer.state.log_history)
    columns = sorted({key for row in trainer.state.log_history for key in row})
    with (output_dir / "log_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(trainer.state.log_history)
    # PEFT сохраняет только адаптер; Trainer также сохраняет токенизатор.
    # Исходная модель остаётся в models/Qwen2.5-1.5B-Instruct.
    model.config.use_cache = True
    trainer.save_model(str(output_dir / "adapter"))
    trainer.save_state()
    save_loss_plot(trainer.state.log_history, output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Адаптер: {output_dir / 'adapter'}\nМетрики обучения: {output_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
