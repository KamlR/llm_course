"""Restore the Qwen3.5 config wrapper required by vLLM's text-only evaluation."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'


def fix_export_config(model_dir):
    model_dir = Path(model_dir)
    base = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    config_path = model_dir / 'config.json'
    original = json.loads((base / 'config.json').read_text())
    current = json.loads(config_path.read_text())
    if current == original:
        print(f'{model_dir}: config already fixed.')
        return
    if current.get('model_type') != 'qwen3_5_text':
        raise ValueError(f'Unexpected model config: {model_dir}')
    # Training changes these runtime settings; restore baseline decoding settings.
    runtime = {'eos_token_id', 'pad_token_id', 'use_cache'}
    for key, value in original['text_config'].items():
        if key not in runtime and current.get(key) != value:
            raise ValueError(f'Architecture mismatch: {key}')
    expected = {}
    for shard in base.glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as weights:
            expected.update({k: weights.get_slice(k).get_shape() for k in weights.keys()
                             if k.startswith('model.language_model.')})
    with safe_open(model_dir / 'model.safetensors', framework='pt', device='cpu') as weights:
        actual = {k: weights.get_slice(k).get_shape() for k in weights.keys()}
    if not expected or actual != expected:
        raise ValueError('Exported text weight names/shapes differ from the base model.')
    backup = model_dir / 'config.text.json'
    if not backup.exists():
        backup.write_bytes(config_path.read_bytes())
    temporary = model_dir / 'config.json.tmp'
    temporary.write_text(json.dumps(original, indent=2) + '\n')
    temporary.replace(config_path)
    print(f'{model_dir}: config fixed; {len(actual)} weight tensors unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dirs', nargs='+', type=Path)
    for model_dir in parser.parse_args().model_dirs:
        fix_export_config(model_dir)
