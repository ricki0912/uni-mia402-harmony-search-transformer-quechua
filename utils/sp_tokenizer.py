"""
Helper para entrenar y cargar un tokenizador SentencePiece (BPE/unigram) una sola vez.
Se usa para subword tokenization en el pipeline HS/entrenamiento.
"""

from pathlib import Path
from typing import Iterable, Tuple

import sentencepiece as spm


def _default_paths(prefix: str = "data/spm_es_qu") -> Tuple[Path, Path, Path]:
    """
    Devuelve paths (corpus, model, vocab) usando un prefijo base.
    """
    base = Path(prefix)
    return base.with_suffix(".txt"), base.with_suffix(".model"), base.with_suffix(".vocab")


def train_sentencepiece(
    corpus_texts: Iterable[str],
    vocab_size: int = 2000,
    model_type: str = "bpe",
    prefix: str = "data/spm_es_qu",
) -> Tuple[Path, Path]:
    """
    Entrena SentencePiece si no existe el modelo. Usa todo el corpus (ES+QU).
    Devuelve (model_path, vocab_path).
    """
    corpus_path, model_path, vocab_path = _default_paths(prefix)
    if model_path.exists() and vocab_path.exists():
        return model_path, vocab_path

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(corpus_texts), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(Path(prefix)),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return model_path, vocab_path


def load_sentencepiece(model_path: Path) -> spm.SentencePieceProcessor:
    """
    Carga un modelo SentencePiece existente.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp
