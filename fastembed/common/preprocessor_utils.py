import json
from typing import Any, Optional
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

from fastembed.image.transform.operators import Compose


def load_special_tokens(model_dir: Path) -> Optional[dict[str, Any]]:
    tokens_map_path = model_dir / "special_tokens_map.json"
    if not tokens_map_path.exists():
        return None

    with open(str(tokens_map_path)) as tokens_map_file:
        tokens_map = json.load(tokens_map_file)

    return tokens_map


def _special_tokens_from_tokenizer(tokenizer: Tokenizer) -> dict[str, int]:
    return {
        added_token.content: token_id
        for token_id, added_token in tokenizer.get_added_tokens_decoder().items()
        if added_token.special
    }


def load_tokenizer(model_dir: Path) -> tuple[Tokenizer, dict[str, int]]:
    config_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(str(config_path)) as config_file:
            config = json.load(config_file)

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise ValueError(f"Could not find tokenizer_config.json in {model_dir}")

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)
        assert "model_max_length" in tokenizer_config or "max_length" in tokenizer_config, (
            "Models without model_max_length or max_length are not supported."
        )
        if "model_max_length" not in tokenizer_config:
            max_context = tokenizer_config["max_length"]
        elif "max_length" not in tokenizer_config:
            max_context = tokenizer_config["model_max_length"]
        else:
            max_context = min(tokenizer_config["model_max_length"], tokenizer_config["max_length"])

    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)
    if not tokenizer.padding:
        pad_token = tokenizer_config.get("pad_token")
        pad_token_id = config.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = tokenizer.token_to_id(pad_token) if pad_token is not None else 0
        tokenizer.enable_padding(pad_id=pad_token_id or 0, pad_token=pad_token or "")

    if tokens_map is None:
        special_token_to_id = _special_tokens_from_tokenizer(tokenizer)
    else:
        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        special_token_to_id = {}

        for token in tokens_map.values():
            if isinstance(token, str):
                special_token_to_id[token] = tokenizer.token_to_id(token)
            elif isinstance(token, dict):
                token_str = token.get("content", "")
                special_token_to_id[token_str] = tokenizer.token_to_id(token_str)

    return tokenizer, special_token_to_id


def load_preprocessor(model_dir: Path) -> Compose:
    preprocessor_config_path = model_dir / "preprocessor_config.json"
    if not preprocessor_config_path.exists():
        raise ValueError(f"Could not find preprocessor_config.json in {model_dir}")

    with open(str(preprocessor_config_path)) as preprocessor_config_file:
        preprocessor_config = json.load(preprocessor_config_file)
        transforms = Compose.from_config(preprocessor_config)
    return transforms
