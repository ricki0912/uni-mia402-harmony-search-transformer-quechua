import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Add repo root to sys.path so we can import config/model/utils
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from model.transformer import Transformer
from model.training import create_masks_factory, limpiar_texto, limpiar_texto_avanzado
from utils.logger import Logger
from utils.sp_tokenizer import load_sentencepiece


START_TOKEN = "<BOS>"
END_TOKEN = "<EOS>"
PADDING_TOKEN = "<PAD>"


def _resolve_checkpoint(path_arg: Optional[str]) -> Path:
    if path_arg:
        ckpt_path = Path(path_arg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No se encontro el checkpoint en {ckpt_path}")
        return ckpt_path

    ckpt_dir = Path(config.CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de checkpoints: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No hay archivos ckpt_*.pt en {ckpt_dir}")
    return ckpts[-1]


def _load_tokenizer() -> Tuple[object, int, int, int, int, Path]:
    spm_prefix = getattr(config, "SPM_PREFIX", f"data/spm_{config.SOURCE_COLUMN}_{config.TARGET_COLUMN}")
    sp_model_path = Path(spm_prefix).with_suffix(".model")
    if not sp_model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo SentencePiece {sp_model_path}. Ejecuta un entrenamiento para generarlo."
        )

    sp = load_sentencepiece(sp_model_path)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
    unk_id = sp.unk_id() if sp.unk_id() >= 0 else 1
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 2
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 3
    return sp, pad_id, bos_id, eos_id, unk_id, sp_model_path


def _build_model(ckpt: dict, tokenizer, pad_id: int, bos_id: int, eos_id: int, unk_id: int, device: torch.device):
    hp = ckpt.get("hp", {})
    vocab_size = ckpt.get("artifacts_meta", {}).get("vocab_size", tokenizer.vocab_size())
    max_len = hp.get("max_sequence_length", ckpt.get("artifacts_meta", {}).get("max_sequence_length", 200))

    source_to_index = {
        PADDING_TOKEN: pad_id,
        START_TOKEN: bos_id,
        END_TOKEN: eos_id,
        "<UNK>": unk_id,
    }

    model = Transformer(
        hp["d_model"],
        2048,
        hp["num_heads"],
        hp["drop_prob"],
        hp["num_layers"],
        max_len,
        vocab_size,
        source_to_index,
        source_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
        tokenizer=tokenizer,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, max_len


def _clean(text: str) -> str:
    return limpiar_texto(limpiar_texto_avanzado(text))


def translate_sentence(
    model: Transformer,
    tokenizer,
    create_masks,
    sentence: str,
    max_len: int,
    pad_id: int,
    eos_id: int,
) -> str:
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    src_batch = [_clean(sentence)]
    decoded_ids = []

    for step in range(max_len):
        tgt_text = tokenizer.decode_ids(decoded_ids) if decoded_ids else ""
        tgt_batch = [tgt_text]
        enc_mask, dec_self_mask, dec_cross_mask = create_masks(src_batch, tgt_batch)

        with torch.no_grad():
            logits = model(
                src_batch,
                tgt_batch,
                encoder_self_attention_mask=enc_mask.to(device),
                decoder_self_attention_mask=dec_self_mask.to(device),
                decoder_cross_attention_mask=dec_cross_mask.to(device),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=False,
            )

        token_pos = min(step, logits.size(1) - 1)
        next_id = int(torch.argmax(logits[0, token_pos, :], dim=-1).item())

        if next_id in (eos_id, pad_id):
            break
        decoded_ids.append(next_id)

    return tokenizer.decode_ids(decoded_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="Traduccion interactiva segun configuracion (SOURCE_LANG -> TARGET_LANG).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al ckpt_*.pt a usar. Si se omite toma el mas reciente en config.CHECKPOINT_DIR.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Limite de tokens generados. Si se omite usa max_sequence_length del checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    logger.print(f"Usando checkpoint: {ckpt_path}")

    tokenizer, pad_id, bos_id, eos_id, unk_id, sp_path = _load_tokenizer()
    logger.print(f"Tokenizador SentencePiece: {sp_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model, hp_max_len = _build_model(ckpt, tokenizer, pad_id, bos_id, eos_id, unk_id, device)
    generate_max_len = args.max_len or hp_max_len
    create_masks = create_masks_factory(generate_max_len, tokenizer, pad_id, bos_id, eos_id)

    print(
        f"\nTraduccion interactiva: {config.SOURCE_LANG} -> {config.TARGET_LANG}"
        f" (max_len={generate_max_len}, device={device})"
    )
    print("Presiona Enter sin texto para salir.\n")

    while True:
        try:
            sentence = input(f"{config.SOURCE_LANG}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if not sentence:
            print("Fin de la sesion.")
            break

        translation = translate_sentence(
            model,
            tokenizer,
            create_masks,
            sentence,
            max_len=generate_max_len,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        print(f"{config.TARGET_LANG}: {translation}\n")


if __name__ == "__main__":
    main()
