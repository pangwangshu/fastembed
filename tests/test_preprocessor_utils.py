import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from fastembed.common.preprocessor_utils import load_special_tokens, load_tokenizer

VOCAB = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5}
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]


def _write_tokenizer_json(model_dir: Path) -> None:
    tokenizer = Tokenizer(WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS])
    tokenizer.save(str(model_dir / "tokenizer.json"))


def _write_tokenizer_config(model_dir: Path) -> None:
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump({"model_max_length": 128, "pad_token": "[PAD]"}, f)


def _write_config(model_dir: Path) -> None:
    with open(model_dir / "config.json", "w") as f:
        json.dump({"pad_token_id": VOCAB["[PAD]"]}, f)


def _write_special_tokens_map(model_dir: Path) -> None:
    with open(model_dir / "special_tokens_map.json", "w") as f:
        json.dump(
            {
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
            },
            f,
        )


def test_load_special_tokens_missing_file_returns_none(tmp_path):
    assert load_special_tokens(tmp_path) is None


def test_load_special_tokens_present_file(tmp_path):
    _write_special_tokens_map(tmp_path)
    tokens_map = load_special_tokens(tmp_path)
    assert tokens_map is not None
    assert tokens_map["pad_token"] == "[PAD]"


def test_load_tokenizer_with_all_files_present(tmp_path):
    _write_tokenizer_json(tmp_path)
    _write_tokenizer_config(tmp_path)
    _write_config(tmp_path)
    _write_special_tokens_map(tmp_path)

    tokenizer, special_token_to_id = load_tokenizer(tmp_path)

    assert special_token_to_id["[PAD]"] == VOCAB["[PAD]"]
    assert special_token_to_id["[CLS]"] == VOCAB["[CLS]"]
    assert tokenizer.padding["pad_id"] == VOCAB["[PAD]"]


def test_load_tokenizer_without_config_json(tmp_path):
    """config.json is optional; pad_token_id should be derived from the tokenizer."""
    _write_tokenizer_json(tmp_path)
    _write_tokenizer_config(tmp_path)
    _write_special_tokens_map(tmp_path)

    tokenizer, special_token_to_id = load_tokenizer(tmp_path)

    assert tokenizer.padding["pad_id"] == VOCAB["[PAD]"]
    assert special_token_to_id["[PAD]"] == VOCAB["[PAD]"]


def test_load_tokenizer_without_special_tokens_map(tmp_path):
    """special_tokens_map.json is optional; special tokens should be derived from the tokenizer."""
    _write_tokenizer_json(tmp_path)
    _write_tokenizer_config(tmp_path)
    _write_config(tmp_path)

    tokenizer, special_token_to_id = load_tokenizer(tmp_path)

    for token in SPECIAL_TOKENS:
        assert special_token_to_id[token] == VOCAB[token]
    assert tokenizer.padding["pad_id"] == VOCAB["[PAD]"]


def test_load_tokenizer_without_config_json_and_special_tokens_map(tmp_path):
    """Both config.json and special_tokens_map.json are optional simultaneously."""
    _write_tokenizer_json(tmp_path)
    _write_tokenizer_config(tmp_path)

    tokenizer, special_token_to_id = load_tokenizer(tmp_path)

    for token in SPECIAL_TOKENS:
        assert special_token_to_id[token] == VOCAB[token]
    assert tokenizer.padding["pad_id"] == VOCAB["[PAD]"]
