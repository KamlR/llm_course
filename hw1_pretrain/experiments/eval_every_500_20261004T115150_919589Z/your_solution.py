import json
import os
from datetime import datetime, timezone
from datasets import Features, Sequence, Value, load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


EXPERIMENT_NAME = 'eval_every_500'

# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/experiments',  # Parent directory for separate runs.
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'save_steps': 500,
    'save_total_limit': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 500,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': True,
    'report_to': 'none',
}


GENERATION_PROMPTS = [
    'Москва — столица России. Город расположен',
    'Солнечная система состоит из',
    'В XIX веке развитие науки',
]
GENERATION_SEED = 42
GENERATION_CONFIG = {
    'max_new_tokens': 128,
    'do_sample': True,
    'temperature': 0.8,
    'top_p': 0.95,
    'top_k': 50,
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


class LossTimingCallback(TrainerCallback):
    """Record wall-clock time for training and evaluation loss logs."""
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time is not None and logs is not None:
            if 'loss' in logs or 'eval_loss' in logs:
                # Trainer appends the log entry before calling on_log.
                state.log_history[-1]['elapsed_seconds'] = time.monotonic() - self.start_time

    def on_train_end(self, args, state, control, **kwargs):
        self.start_time = None


def prepare_tokenizer():
    """Load the tokenizer and use EOS as the padding token."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    """Tokenize a batch of texts and mask padding positions in labels."""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_attention_mask=True,
    )
    labels = [
        [token_id if mask else -100 for token_id, mask in zip(input_ids, attention_mask)]
        for input_ids, attention_mask in zip(tokenized[INPUT_IDS], tokenized[ATTENTION_MASK])
    ]
    return {
        INPUT_IDS: tokenized[INPUT_IDS],
        ATTENTION_MASK: tokenized[ATTENTION_MASK],
        LABELS: labels,
    }


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """Save the dataset as parquet shards while preserving row order."""
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard_path = os.path.join(output_dir, f'{index:05d}.parquet')
        shard.to_parquet(shard_path)


def prepare_dataset():
    """Load Russian Wikipedia, tokenize its articles, and save parquet shards."""
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=min(4, os.cpu_count() or 1),
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=dataset.column_names,
        features=Features({
            INPUT_IDS: Sequence(Value('int32')),
            ATTENTION_MASK: Sequence(Value('int8')),
            LABELS: Sequence(Value('int32')),
        }),
        desc='Tokenizing Wikipedia',
    )
    save_as_parquets(tokenized_dataset)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """Load parquet shards in filename order and return the combined dataset."""
    parquet_files = [
        os.path.join(data_dir, filename)
        for filename in sorted(os.listdir(data_dir))
        if filename.endswith('.parquet')
        and os.path.isfile(os.path.join(data_dir, filename))
    ]
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {data_dir}. Run prepare_dataset() first.')
    dataset = load_dataset('parquet', data_files=parquet_files)
    return dataset['train']


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def plot_loss(log_history, output_dir):
    """Save training-time loss curves by optimizer step and elapsed time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for metric, label, color in [
        ('loss', 'Train loss', 'tab:blue'),
        ('eval_loss', 'Eval loss', 'tab:orange'),
    ]:
        records = [entry for entry in log_history if metric in entry]
        if not records:
            continue
        axes[0].plot(
            [entry['step'] for entry in records],
            [entry[metric] for entry in records],
            label=label, color=color, marker='o' if metric == 'eval_loss' else None,
        )
        timed_records = [entry for entry in records if 'elapsed_seconds' in entry]
        if timed_records:
            axes[1].plot(
                [entry['elapsed_seconds'] / 60 for entry in timed_records],
                [entry[metric] for entry in timed_records],
                label=label, color=color, marker='o' if metric == 'eval_loss' else None,
            )

    for ax, xlabel in zip(axes, ['Optimizer step', 'Elapsed time (minutes)']):
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()
    fig.suptitle('Pretraining loss (time includes evaluation and checkpoint saving)')
    plot_path = os.path.join(output_dir, 'loss.png')
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f'Loss plot saved to {plot_path}')


def save_generations(model, tokenizer, output_dir):
    """Generate continuations for fixed prompts and save readable and JSON results."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    samples = []
    with torch.inference_mode():
        for index, prompt in enumerate(GENERATION_PROMPTS):
            seed = GENERATION_SEED + index
            set_seed(seed)
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            output_ids = model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            new_ids = output_ids[0, inputs[INPUT_IDS].shape[1]:]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
            samples.append({
                'prompt': prompt,
                'seed': seed,
                'continuation': continuation,
                'full_text': prompt + continuation,
                'generated_tokens': len(new_ids),
            })

    with open(os.path.join(output_dir, 'generations.json'), 'w', encoding='utf-8') as file:
        json.dump({'generation_config': GENERATION_CONFIG, 'samples': samples},
                  file, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'generations.txt'), 'w', encoding='utf-8') as file:
        for index, sample in enumerate(samples, start=1):
            file.write(f"Example {index} (seed={sample['seed']})\n")
            file.write(f"Prompt: {sample['prompt']}\n")
            file.write(f"Continuation: {sample['continuation']}\n\n")
            print(f"\nGeneration {index}:\n{sample['full_text']}")
    return samples


def create_experiment_dir():
    """Create a new run directory without overwriting earlier experiments."""
    name = ''.join(char if char.isalnum() or char in '-_' else '_' for char in EXPERIMENT_NAME)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output_dir = os.path.join(TRAINING_CONFIG['output_dir'], f'{name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def save_report(trainer, training_history, train_metrics, eval_results, samples):
    """Write experiment facts and leave hypothesis and conclusions for manual analysis."""
    output_dir = trainer.args.output_dir
    eval_records = [entry for entry in training_history if 'eval_loss' in entry]
    best_record = min(eval_records, key=lambda entry: entry['eval_loss']) if eval_records else None
    best_checkpoint = trainer.state.best_model_checkpoint
    checkpoint_name = os.path.basename(os.path.normpath(best_checkpoint)) if best_checkpoint else None
    selected_model = (
        f'лучший сохранённый checkpoint `{checkpoint_name}`'
        if trainer.args.load_best_model_at_end and checkpoint_name
        else 'последние веса обучения'
    )
    summary = {
        'experiment_name': EXPERIMENT_NAME,
        'best_eval_loss': best_record['eval_loss'] if best_record else None,
        'best_eval_step': best_record['step'] if best_record else None,
        'best_checkpoint': checkpoint_name,
        'final_eval_loss': eval_results['eval_loss'],
        'evaluated_model': selected_model,
        'optimizer_steps': trainer.state.global_step,
        'train_runtime_seconds': train_metrics.get('train_runtime'),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    effective_config = json.loads(trainer.args.to_json_string())
    config_keys = list(TRAINING_CONFIG) + ['lr_scheduler_type', 'seed', 'per_device_eval_batch_size']
    report_config = {key: effective_config[key] for key in config_keys if key != 'output_dir'}
    lines = [
        f'# Эксперимент: {EXPERIMENT_NAME}', '',
        f'Папка запуска: `{os.path.basename(output_dir)}` (время в имени — UTC).', '',
        '## Описание эксперимента', '',
        '**Гипотеза:** _Заполним вручную._', '',
        '**Что меняем и с каким запуском сравниваем:** _Заполним вручную._', '',
        'Модель: Qwen3 из заданного конфига, обучение с нуля.',
        'Датасет: `wikimedia/wikipedia`, `20231101.ru`; первые 5000 статей — evaluation.',
        f'Токенизатор: `{TOKENIZER_NAME}`. Длина последовательности: {MAX_LENGTH}.',
        f'Лимит обучения: {MAX_TRAINING_TIME_SECONDS} секунд. Веса создаются в BF16.', '',
        '### Настройки обучения', '',
        '```json', json.dumps(report_config, ensure_ascii=False, indent=2), '```', '',
        '## Результаты и выводы', '',
        '### Метрики', '',
        '| Метрика | Значение |', '|---|---|',
        f"| Лучший eval_loss за время обучения | {summary['best_eval_loss']} |",
        f"| Шаг лучшего eval_loss | {summary['best_eval_step']} |",
        f"| Финальный eval_loss выбранной модели | {summary['final_eval_loss']} |",
        f"| Число шагов оптимизатора | {summary['optimizer_steps']} |",
        f"| Время trainer.train(), секунд | {summary['train_runtime_seconds']} |", '',
        f'Для финальной оценки и генераций использован {selected_model}.',
        'Его веса и токенизатор сохранены в этой папке.',
        'Время включает промежуточные оценки и сохранения; остановка проверяется после шага.',
        'Финальная оценка, график и генерации выполняются после trainer.train().', '',
        '### График loss', '',
        '![Train и eval loss по шагам и времени](loss.png)', '',
        'Кривая отражает оценки во время обучения. Повторная оценка выбранных весов в неё не включена.', '',
        '### Генерации', '',
        '```json', json.dumps(GENERATION_CONFIG, ensure_ascii=False, indent=2), '```', '',
    ]
    for index, sample in enumerate(samples, start=1):
        lines.extend([f"#### Пример {index} (seed={sample['seed']})", '', '**Промпт:**', ''])
        lines.extend('    ' + line for line in sample['prompt'].splitlines())
        lines.extend(['', '**Продолжение:**', ''])
        lines.extend('    ' + line for line in (sample['continuation'] or '[Пустое продолжение]').splitlines())
        lines.append('')
    lines.extend([
        '### Выводы', '',
        '_Заполним вручную: подтвердилась ли гипотеза, сравнение с другими запусками, '
        'наблюдения по loss и генерациям, следующий эксперимент._', '',
        '### Файлы результатов', '',
        '- [Сводные метрики](results.json)',
        '- [Финальная оценка](eval_results.json)',
        '- [Метрики обучения](train_results.json)',
        '- [История метрик и лучший checkpoint](trainer_state.json)',
        '- [Генерации TXT](generations.txt) и [JSON](generations.json)', '',
    ])
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'x', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    print(f'Experiment report saved to {report_path}')


def train_model():
    """Train from scratch, evaluate, and save the model and metric history."""
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)

    output_dir = create_experiment_dir()
    print(f'Experiment output directory: {output_dir}')
    training_args = TrainingArguments(**{**TRAINING_CONFIG, 'output_dir': output_dir})
    with open(os.path.join(training_args.output_dir, 'training_config.json'), 'w', encoding='utf-8') as file:
        file.write(training_args.to_json_string())
    set_seed(training_args.seed)
    model = create_model(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS), # dont change
            LossTimingCallback(),
        ]
    )
    train_results = trainer.train()
    trainer.save_metrics('train', train_results.metrics)
    # Exclude the later re-evaluation of the selected checkpoint from the curves.
    training_history = list(trainer.state.log_history)

    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_metrics('eval', eval_results)
    trainer.save_state()
    trainer.save_model()
    plot_loss(training_history, training_args.output_dir)
    samples = save_generations(trainer.model, tokenizer, training_args.output_dir)
    save_report(trainer, training_history, train_results.metrics, eval_results, samples)


if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()
    
    # Step 2: Train the model
    train_model()
