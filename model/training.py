
## importaciòn de datos desde un archivo excel con pandas
import pandas as pd
import re
import unicodedata
from pathlib import Path
from model.transformer import Transformer # this is the transformer.py file
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from utils.logger import Logger
from utils.sp_tokenizer import load_sentencepiece, train_sentencepiece
import time
NEG_INFTY = -1e9

def load_clean_dataframe():
    ruta_archivo_excel = 'data/combined_dataframe_v4.xlsx'

    hojas = ['Dennis', 'Ayrton', 'Ruddy', 'Ricardo', 'Jose']

    df_list = []

    for hoja in hojas:
        df_temp = pd.read_excel(ruta_archivo_excel, sheet_name=hoja)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    print("Excel file loaded successfully with added columns 'es,' and 'qu'.")

    Logger.print("Aplicar codificación UTF-8 a las columnas de texto")

    for col in df.select_dtypes(include=[object]):
        df[col] = df[col].apply(lambda x: x.encode('utf-8').decode('utf-8') if isinstance(x, str) else x)

    df = df.dropna(how='any')

    print(df[['es', 'qu']].head())

    df = df.dropna(subset=["es", "qu"])

    df["len_es"] = df["es"].astype(str).apply(len)
    df["len_qu"] = df["qu"].astype(str).apply(len)
    df["abs_diff"] = (df["len_es"] - df["len_qu"]).abs()

    condicion_original = (
        (df["abs_diff"] > 150) |
        (
            ((df["len_es"] <= 40) | (df["len_qu"] <= 40)) &
            (df["abs_diff"] > 40)
        )
    )

    filtered_df = df[~condicion_original]

    filtered_df = filtered_df[["es", "qu"]]

    print(filtered_df)
    ruta_archivo = 'data/data_preprocesada_sin_outliers_v4_biblia.xlsx'
    filtered_df.to_excel(ruta_archivo, index=False)

    print(f"DataFrame guardado en {ruta_archivo}")

    return filtered_df
def load_dataframeFromXLSX(ruta_archivo_excel):
    #ruta_archivo_excel = 'data/data_preprocesada_sin_outliers_v4_biblia.xlsx'
    Logger.print(f"load_dataframeFromXLSX()=> Cargando archivo Excel desde {ruta_archivo_excel}")
    df = pd.read_excel(ruta_archivo_excel)
    #print("Excel file loaded successfully.")
    return df

def limpiar_texto_avanzado(texto):
    if pd.isnull(texto):
        return ''
    texto = unicodedata.normalize('NFKC', texto)
    texto = re.sub(r'[^\w\s\'-]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    texto = texto.lower()
    return texto

def limpiar_texto(texto):
    if pd.isnull(texto):
        return ""
    return re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s.,!?¿¡:;\"'\-()[\]{}]", "", texto)


def sort_key(char):
        if char.isdigit():
            return (0, char)  # Numbers first
        elif char.isalpha():
            return (1, char.lower())  # Letters next, case insensitive
        else:
            return (2, char)

def extract_unique_tokens(text_list):
  """Extracts unique tokens (characters) from a list of texts."""
  all_tokens = set()
  for text in text_list:
    for char in text:
      all_tokens.add(char)
  tokens= list(all_tokens)
  tokens.sort()
  #return tokens
  return sorted(tokens, key=sort_key)


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space



def _encode_sentence(
    sentence: str,
    tokenizer,
    max_sequence_length: int,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    pad_id: int,
):
    ids = tokenizer.encode(sentence, out_type=int, add_bos=False, add_eos=False)
    if add_bos:
        ids = [bos_id] + ids
    if add_eos:
        ids = ids + [eos_id]
    ids = ids[:max_sequence_length]
    if len(ids) < max_sequence_length:
        ids = ids + [pad_id] * (max_sequence_length - len(ids))
    return ids


def create_masks_factory(max_sequence_length, tokenizer, pad_id, bos_id, eos_id):
    """
    Crea mascaras usando longitudes de subwords (SentencePiece).
    """
    def create_masks(eng_batch, kn_batch):
        num_sentences = len(eng_batch)
        look_ahead_mask = torch.triu(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool), diagonal=1)
        encoder_padding_mask = torch.zeros((num_sentences, max_sequence_length, max_sequence_length), dtype=torch.bool)
        decoder_padding_mask_self_attention = torch.zeros_like(encoder_padding_mask)
        decoder_padding_mask_cross_attention = torch.zeros_like(encoder_padding_mask)

        for idx in range(num_sentences):
            src_ids = tokenizer.encode(eng_batch[idx], out_type=int, add_bos=False, add_eos=False)
            tgt_ids = tokenizer.encode(kn_batch[idx], out_type=int, add_bos=False, add_eos=False)
            src_len = min(len(src_ids), max_sequence_length)
            tgt_len = min(len(tgt_ids) + 2, max_sequence_length)  # +BOS +EOS en decoder

            src_pad_positions = np.arange(src_len, max_sequence_length)
            tgt_pad_positions = np.arange(tgt_len, max_sequence_length)

            encoder_padding_mask[idx, :, src_pad_positions] = True
            encoder_padding_mask[idx, src_pad_positions, :] = True

            decoder_padding_mask_self_attention[idx, :, tgt_pad_positions] = True
            decoder_padding_mask_self_attention[idx, tgt_pad_positions, :] = True

            decoder_padding_mask_cross_attention[idx, :, src_pad_positions] = True
            decoder_padding_mask_cross_attention[idx, tgt_pad_positions, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
        decoder_self_attention_mask = torch.where(look_ahead_mask | decoder_padding_mask_self_attention, NEG_INFTY, 0)
        decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

    return create_masks

class TextDataset(Dataset):

    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.kannada_sentences[idx]

    




def build_training_artifacts(max_sequence_length: int = 200, dataset_path: str = None) -> dict:
    Logger.print("Version Torch:", torch.__version__)
    Logger.print("Version CUDA en Torch:", torch.version.cuda)
    Logger.print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        Logger.print("GPU detectada:", torch.cuda.get_device_name(0))
    else:
        Logger.print("No se detecta GPU CUDA.")

    df = load_dataframeFromXLSX(dataset_path)
    df["es"] = df["es"].astype(str).apply(limpiar_texto_avanzado).apply(limpiar_texto)
    df["qu"] = df["qu"].astype(str).apply(limpiar_texto_avanzado).apply(limpiar_texto)
    df = df.dropna(subset=["es", "qu"], how="any")

    source_language_array = df["es"].tolist()
    targe_language_array = df["qu"].tolist()
    name_source_lang = "Spanish"
    name_target_lang = "Quechua"
    Logger.print(f"Traduccion de {name_source_lang} a {name_target_lang}")

    START_TOKEN = "<BOS>"
    PADDING_TOKEN = "<PAD>"
    END_TOKEN = "<EOS>"

    corpus_texts = source_language_array + targe_language_array
    sp_model_path, _ = train_sentencepiece(
        corpus_texts=corpus_texts,
        vocab_size=2000,
        model_type="bpe",
        prefix="data/spm_es_qu",
    )
    sp = load_sentencepiece(sp_model_path)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
    unk_id = sp.unk_id() if sp.unk_id() >= 0 else 1
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 2
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 3

    source_language_sentences = source_language_array
    target_language_sentences = targe_language_array

    target_language_vocabulary = None
    source_to_index = {PADDING_TOKEN: pad_id, START_TOKEN: bos_id, END_TOKEN: eos_id, "<UNK>": unk_id}
    target_to_index = source_to_index.copy()
    index_to_target = {v: k for k, v in target_to_index.items()}
    index_to_source = index_to_target

    dataset = TextDataset(source_language_sentences, target_language_sentences)
    Logger.print(f"Oraciones cargadas: {len(source_language_sentences)} | Tokenizer SP: {sp_model_path}")

    return {
        "dataset": dataset,
        "create_masks": create_masks_factory(max_sequence_length, sp, pad_id, bos_id, eos_id),
        "index_to_target": index_to_target,
        "target_to_index": target_to_index,
        "index_to_source": index_to_source,
        "source_to_index": source_to_index,
        "START_TOKEN": START_TOKEN,
        "END_TOKEN": END_TOKEN,
        "PADDING_TOKEN": PADDING_TOKEN,
        "max_sequence_length": max_sequence_length,
        "vocab_size": sp.vocab_size(),
        "name_source_lang": name_source_lang,
        "name_target_lang": name_target_lang,
        "target_language_vocabulary": target_language_vocabulary,
        "tokenizer": sp,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unk_id": unk_id,
        "sp_model_path": str(sp_model_path),
    }

def train_single_run(hp, artifacts, resume_from=None, checkpoint_dir="checkpoints", checkpoint_every=5):
    start_time = time.time()
    Logger.print("Iniciando entrenamiento con los siguientes hiperparámetros:")
    Logger.print(hp)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    # Transformer Hyperparameters
    d_model = hp["d_model"]
    batch_size = hp["batch_size"]
    ffn_hidden = hp["ffn_hidden"]
    num_heads = hp["num_heads"]
    drop_prob = hp["drop_prob"]
    num_layers = hp["num_layers"]
    max_sequence_length = hp["max_sequence_length"]
    num_epochs = hp["epochs"]
    lr = hp["lr"]
    #nuevos parametros
    weight_decay = float(hp.get("weight_decay", 0.0))
    label_smoothing = float(hp.get("label_smoothing", 0.0))
    warmup_steps = int(hp.get("warmup_steps", 400))
    grad_clip = float(hp.get("grad_clip", 1.0))



    
    vocab_size = artifacts["vocab_size"]
    source_to_index = artifacts["source_to_index"]
    target_to_index = artifacts["target_to_index"]
    START_TOKEN = artifacts["START_TOKEN"]
    END_TOKEN = artifacts["END_TOKEN"]
    PADDING_TOKEN = artifacts["PADDING_TOKEN"]
    dataset = artifacts["dataset"]
    create_masks = artifacts["create_masks"]
    tokenizer = artifacts["tokenizer"]
    pad_id = artifacts["pad_id"]
    bos_id = artifacts["bos_id"]
    eos_id = artifacts["eos_id"]
    name_source_lang = artifacts["name_source_lang"]
    name_target_lang = artifacts["name_target_lang"]

    transformer = Transformer(
        d_model,
        ffn_hidden,
        num_heads,
        drop_prob,
        num_layers,
        max_sequence_length,
        vocab_size,
        source_to_index,
        target_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
        tokenizer=tokenizer,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    #dataset = TextDataset(source_language_sentences, target_language_sentences)

    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    iterator = iter(train_loader)

    # Definir la función de pérdida y el optimizador
    criterian = nn.CrossEntropyLoss(
        ignore_index=target_to_index[PADDING_TOKEN],
        label_smoothing=label_smoothing,
        reduction="none",
    )
        


    # When computing the loss, we are ignoring cases when the label is the padding token
    for params in transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    #optim = torch.optim.Adam(transformer.parameters(), lr=lr)
    optim = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay)
    global_step = 0

    def lr_lambda(step):
        # warmup lineal hasta warmup_steps y luego constante
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-8, step / float(warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    


    #ENTRENAMIENTO DEL MODELO
    #AGREGAMOS METRICAS
    smoothing = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    losses = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    train_losses = []

    # Reanudar desde checkpoint si se provee
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        transformer.load_state_dict(ckpt.get("model_state", {}))
        optim.load_state_dict(ckpt.get("optimizer_state", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        m = ckpt.get("metrics", {})
        losses = m.get("loss_history", losses)
        bleu_scores = m.get("bleu_history", bleu_scores)
        rouge_hist = m.get("rouge", {})
        rouge1_scores = rouge_hist.get("rouge1_history", rouge1_scores)
        rouge2_scores = rouge_hist.get("rouge2_history", rouge2_scores)
        rougeL_scores = rouge_hist.get("rougeL_history", rougeL_scores)

    transformer.train()
    transformer.to(device)
    total_loss = 0

    def decode_tokens_to_text(token_ids):
        filtered = []
        for idx in token_ids:
            if idx in (pad_id, bos_id):
                continue
            if idx == eos_id:
                break
            filtered.append(int(idx))
        return tokenizer.decode_ids(filtered)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch}")
        iterator = iter(train_loader)

    #AGREGAMOS METRICAS
        total_loss = 0
        total_bleu_score = 0
        total_rouge_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

        for batch_num, batch in enumerate(iterator):
            transformer.train()
            eng_batch, kn_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, kn_batch)
            optim.zero_grad()
            kn_predictions = transformer(eng_batch,
                                        kn_batch,
                                        encoder_self_attention_mask.to(device),
                                        decoder_self_attention_mask.to(device),
                                        decoder_cross_attention_mask.to(device),
                                        enc_start_token=False,
                                        enc_end_token=False,
                                        dec_start_token=True,
                                        dec_end_token=True)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
            loss = criterian(
                kn_predictions.view(-1, vocab_size).to(device),
                labels.view(-1).to(device)
            ).to(device)
            #valid_indicies = torch.where(labels.view(-1) == target_to_index[PADDING_TOKEN], False, True)
            #loss = loss.sum() / valid_indicies.sum()
            mask = (labels.view(-1).to(device) != target_to_index[PADDING_TOKEN]).float()
            loss = (loss * mask).sum() / mask.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=grad_clip)

            optim.step()
            scheduler.step()      # <-- IMPORTANTÍSIMO: después del step
            global_step += 1

            total_loss += loss.item()
            #train_losses.append(loss.item())

            #codigo agregado
            # Calcular BLEU y ROUGE para la prediccion actual
            kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1).tolist()
            predicted_sentence = decode_tokens_to_text(kn_sentence_predicted)

            reference_sentence = kn_batch[0]
            bleu_score = sentence_bleu([reference_sentence.split()], predicted_sentence.split(), smoothing_function=smoothing)
            total_bleu_score += bleu_score

            rouge_scores = scorer.score(reference_sentence, predicted_sentence)
            for key in total_rouge_score:
                total_rouge_score[key] += rouge_scores[key].fmeasure


            #DUDA
            if batch_num % 1000 == 0:
                print(f"Iteration {batch_num} : {loss.item()}")
                print(f"{name_source_lang}: {eng_batch[0]}")
                print(f"{name_target_lang} Translation: {kn_batch[0]}")
                kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1).tolist()
                predicted_sentence = decode_tokens_to_text(kn_sentence_predicted)
                print(f"{name_target_lang} Prediction: {predicted_sentence}")

        # Promedio de las métricas de la época
        avg_loss = total_loss / len(train_loader)
        avg_bleu_score = total_bleu_score / len(train_loader)
        avg_rouge_score = {k: v / len(train_loader) for k, v in total_rouge_score.items()}

        # Guardar las métricas en listas para graficar después
        losses.append(avg_loss)
        bleu_scores.append(avg_bleu_score)
        rouge1_scores.append(avg_rouge_score['rouge1'])
        rouge2_scores.append(avg_rouge_score['rouge2'])
        rougeL_scores.append(avg_rouge_score['rougeL'])

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        Logger.print(f"[Train] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}, BLEU: {avg_bleu_score:.6f} | t_epoch={epoch_time:.1f}s, t_total={total_elapsed:.1f}s")
        Logger.print(
            f"[Train] ROUGE-1: {avg_rouge_score['rouge1']:.6f}, ROUGE-2: {avg_rouge_score['rouge2']:.6f}, ROUGE-L: {avg_rouge_score['rougeL']:.6f}"
        )

        # Guardar checkpoint de progreso
        if ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == num_epochs):
            ckpt_path = ckpt_dir / f"ckpt_{int(time.time()*1000)}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": transformer.state_dict(),
                "optimizer_state": optim.state_dict(),
                "hp": hp,
                "metrics": {
                    "loss_history": losses,
                    "bleu_history": bleu_scores,
                    "rouge": {
                        "rouge1_history": rouge1_scores,
                        "rouge2_history": rouge2_scores,
                        "rougeL_history": rougeL_scores,
                    },
                },
                "artifacts_meta": {
                    "max_sequence_length": artifacts.get("max_sequence_length"),
                    "vocab_size": artifacts.get("vocab_size"),
                },
            }, ckpt_path)
            Logger.print(f"Checkpoint guardado: {ckpt_path}")


    #
    #GUARDAR EL MODELO
    # torch.save(transformer.state_dict(), 'transformer_model_3_capas_'+str(num_epochs)+'epochs'+'.pth')
    #GUARDAR EL MODELO
    #from google.colab import drive
    #import os
    #drive.mount('/content/drive')
    #save_path = '/content/drive/MyDrive/transformer_model'  # Change this to your desired path

    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)

    #torch.save(transformer.state_dict(), os.path.join(save_path, 'transformer_model_3_capas'+str(len(dataset))+'.pth'))

    # Download the saved model (optional)
    #from google.colab import files
    #files.download(os.path.join(save_path, 'transformer_model.pth'))


    #PRESENTAR METRICAS

    # Graficar las métricas después de finalizar el entrenamiento
    #plt.figure(figsize=(12, 6))

    # Gráfico de la pérdida
    """plt.subplot(2, 2, 1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    """

    # Gráfico de BLEU
    """plt.subplot(2, 2, 2)
    plt.plot(bleu_scores, label="BLEU Score", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score over Epochs")
    plt.legend()
    """
    # Gráfico de ROUGE-1, ROUGE-2, ROUGE-L
    """plt.subplot(2, 2, 3)
    plt.plot(rouge1_scores, label="ROUGE-1", color="green")
    plt.plot(rouge2_scores, label="ROUGE-2", color="blue")
    plt.plot(rougeL_scores, label="ROUGE-L", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("ROUGE Score")
    plt.title("ROUGE Scores over Epochs")
    plt.legend()"""

    """
    plt.tight_layout()
    plt.show()"""

    
    """
    metrics = {
        "loss": losses[-1],
        "bleu": bleu_scores[-1],
        "rouge": {
            "rouge1": rouge1_scores[-1],
            "rouge2": rouge2_scores[-1],
            "rougeL": rougeL_scores[-1],
        },
        "epochs_run": num_epochs,
    }
    """
    end_time = time.time()
    training_time = end_time - start_time
    metrics = {
        "loss_history": losses,  # Lista completa de pérdidas
        "bleu_history": bleu_scores,  # Lista completa de BLEU scores
        "rouge": {
            "rouge1_history": rouge1_scores,  # Lista completa de ROUGE-1
            "rouge2_history": rouge2_scores,  # Lista completa de ROUGE-2
            "rougeL_history": rougeL_scores,  # Lista completa de ROUGE-L
        },
        # También mantener los valores finales para referencia rápida
        "final_metrics": {
            "loss": losses[-1],
            "bleu": bleu_scores[-1],
            "rouge1": rouge1_scores[-1],
            "rouge2": rouge2_scores[-1],
            "rougeL": rougeL_scores[-1],
        },
        "epochs_run": num_epochs,
        # Añadir estadísticas adicionales
        "stats": {
            "min_loss": min(losses),
            "max_bleu": max(bleu_scores),
            "best_epoch_loss": losses.index(min(losses)),
            "best_epoch_bleu": bleu_scores.index(max(bleu_scores)),
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "training_time_hours": training_time / 3600,
            "avg_time_per_epoch": training_time / num_epochs
        }, 
        "bleu": bleu_scores[-1]
    }
    
    return metrics


if __name__ == "__main__":
    artifacts = build_training_artifacts(max_sequence_length=200)
    hp = {
        "d_model": 512 , 
        "batch_size": 30 ,
        "ffn_hidden": 2048 , 
        "num_heads": 8, 
        "drop_prob": 0.1 , 
        "num_layers": 3 , 
        "max_sequence_length": 200,
    }
    train_single_run(hp, artifacts, epochs=100)
