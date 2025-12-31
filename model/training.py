
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



def create_masks_factory(max_sequence_length):
    def create_masks(eng_batch, kn_batch):
        num_sentences = len(eng_batch)
        look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
        encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
        decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
        decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

        for idx in range(num_sentences):
            eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
            eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
            kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
            encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
            encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
            decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
            decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
            decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
        decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
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

    


def build_training_artifacts(max_sequence_length: int = 200, dataset_path:str=None) -> dict:
    Logger.print("Versión Torch:", torch.__version__)
    Logger.print("Versión CUDA en Torch:", torch.version.cuda)
    Logger.print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        Logger.print("GPU detectada:", torch.cuda.get_device_name(0))
    else:
        Logger.print("No se detecta GPU CUDA.")
    ##df=load_clean_dataframe()
    df=load_dataframeFromXLSX(dataset_path)

    #print(limpiar_texto_avanzado("asd © - () {}[a].. 1234 PËÉRRO. "))
    df['es'] = df['es'].astype(str).apply(limpiar_texto_avanzado)
    df['qu'] = df['qu'].astype(str).apply(limpiar_texto_avanzado)


    df['es'] = df['es'].apply(limpiar_texto)
    df['qu'] = df['qu'].apply(limpiar_texto)

    df = df.dropna(subset=['es', 'qu'], how='any')
    #print(df[['es', 'qu']].head())


    source_language_array = df['es'].tolist()
    targe_language_array = df['qu'].tolist()
    name_source_lang = 'Spanish'
    name_target_lang = 'Quechua'
    Logger.print("Traducción de {} a {}".format(name_source_lang, name_target_lang))

    #print(f"{name_source_lang} Array:")
    #print(source_language_array[:5])  # Mostrar los primeros 5 elementos del array

    #print(f"{name_target_lang} Array:")
    #print(targe_language_array[:5])  # Mostrar los primeros 100 caracteres del JSON


    START_TOKEN = '<START>'
    PADDING_TOKEN = '<PADDING>'
    END_TOKEN = '<END>'



    source_language_tokens = extract_unique_tokens(source_language_array)
    target_language_tokens = extract_unique_tokens(targe_language_array)

    source_language_tokens.insert(0,'Ll')
    source_language_tokens.insert(0,'Ch')
    source_language_tokens.insert(0,'ll')
    source_language_tokens.insert(0,'ch')

    target_language_tokens.insert(0, 'Ll')
    target_language_tokens.insert(0, 'Ch')
    target_language_tokens.insert(0, 'll')
    target_language_tokens.insert(0, 'ch')


    source_language_tokens.insert(0,START_TOKEN)
    target_language_tokens.insert(0, START_TOKEN)


    source_language_tokens.append(PADDING_TOKEN)
    source_language_tokens.append(END_TOKEN)

    target_language_tokens.append(PADDING_TOKEN)
    target_language_tokens.append(END_TOKEN)

    #print(f"{name_source_lang} Tokens:")
    #print(source_language_tokens)

    #print(f"\n{name_target_lang} Tokens:")
    #print(target_language_tokens)

    source_language_vocabulary = source_language_tokens
    target_language_vocabulary = target_language_tokens

    #print(f"{name_source_lang} Sentences:")
    source_language_sentences = source_language_array

    #print(f"{name_target_lang} Sentences")
    target_language_sentences = targe_language_array

    #print(source_language_sentences[:5])
    #print(target_language_sentences[:5])

    #print(len(source_language_sentences))
    #print(len(target_language_sentences))



    index_to_target = {k:v for k,v in enumerate(target_language_vocabulary)}
    target_to_index = {v:k for k,v in enumerate(target_language_vocabulary)}
    index_to_source = {k:v for k,v in enumerate(source_language_vocabulary)}
    source_to_index = {v:k for k,v in enumerate(source_language_vocabulary)}

    source_language_sentences[:10]

    PERCENTILE = 97
    #print( f"{PERCENTILE}th percentile length {name_source_lang}: {np.percentile([len(x) for x in target_language_sentences], PERCENTILE)}" )
    #print( f"{PERCENTILE}th percentile length {name_target_lang}: {np.percentile([len(x) for x in source_language_sentences], PERCENTILE)}" )


    #max_sequence_length = 200

    valid_sentence_indicies = []
    for index in range(len(target_language_sentences)):
        kannada_sentence, english_sentence = target_language_sentences[index], source_language_sentences[index]
        if is_valid_length(kannada_sentence, max_sequence_length) \
        and is_valid_length(english_sentence, max_sequence_length) \
        and is_valid_tokens(kannada_sentence, target_language_vocabulary):
            valid_sentence_indicies.append(index)

    #print(f"Number of sentences: {len(target_language_sentences)}")
    #print(f"Number of valid sentences: {len(valid_sentence_indicies)}")
    Logger.print("Numero de oraciones antes de filtrar por longitud y vocabulario: {}".format(len(target_language_sentences)))
    Logger.print("Numero de oraciones válidas después de filtrar por longitud y vocabulario: {}".format(len(valid_sentence_indicies)))

    target_language_sentences = [target_language_sentences[i] for i in valid_sentence_indicies]
    source_language_sentences = [source_language_sentences[i] for i in valid_sentence_indicies]

    #print(len(source_language_sentences))
    #print(len(target_language_sentences))

    dataset = TextDataset(source_language_sentences, target_language_sentences)
    return {
        "dataset": dataset,
        "create_masks": create_masks_factory(max_sequence_length),
        "index_to_target": index_to_target,
        "target_to_index": target_to_index,
        "index_to_source": index_to_source,
        "source_to_index": source_to_index,
        "START_TOKEN": START_TOKEN,
        "END_TOKEN": END_TOKEN,
        "PADDING_TOKEN": PADDING_TOKEN,
        "max_sequence_length": max_sequence_length,
        "vocab_size": len(index_to_target),
        "name_source_lang": name_source_lang,
        "name_target_lang": name_target_lang,
        "target_language_vocabulary": target_language_vocabulary, 
    }



def train_single_run(hp, artifacts, resume_from=None, checkpoint_dir="checkpoints", checkpoint_every=5):
    start_time = time.time()
    Logger.print("Iniciando entrenamiento con los siguientes hiperparámetros:")
    Logger.print(hp)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    # Transformer Hyperparameters
    d_model = hp["d_model"] #
    batch_size = hp["batch_size"] #
    ffn_hidden = hp["ffn_hidden"] #
    num_heads = hp["num_heads"] #
    drop_prob = hp["drop_prob"] #
    num_layers = hp["num_layers"] #
    max_sequence_length = hp["max_sequence_length"]
    #num_epochs = int(epochs)
    num_layers = hp["num_layers"] #
    num_epochs = hp["epochs"]
    lr = hp["lr"]


    
    #d_model = 512 #
    #batch_size = 30 #
    #ffn_hidden = 2048 #
    #num_heads = 8 #
    #drop_prob = 0.1 #
    #num_layers = 3 #
    #max_sequence_length = 200 #
    

    #kn_vocab_size = len(target_language_vocabulary)
    kn_vocab_size = len(artifacts["target_language_vocabulary"])
    source_to_index = artifacts["source_to_index"]
    target_to_index = artifacts["target_to_index"]
    START_TOKEN = artifacts["START_TOKEN"]
    END_TOKEN = artifacts["END_TOKEN"]
    PADDING_TOKEN = artifacts["PADDING_TOKEN"]
    dataset = artifacts["dataset"]
    create_masks = artifacts["create_masks"]
    index_to_target = artifacts["index_to_target"]
    name_source_lang = artifacts["name_source_lang"]
    name_target_lang = artifacts["name_target_lang"]

    transformer = Transformer(d_model,
                            ffn_hidden,
                            num_heads,
                            drop_prob,
                            num_layers,
                            max_sequence_length,
                            kn_vocab_size,
                            source_to_index,
                            target_to_index,
                            START_TOKEN,
                            END_TOKEN,
                            PADDING_TOKEN)

    #dataset = TextDataset(source_language_sentences, target_language_sentences)

    train_loader = DataLoader(dataset, batch_size)
    iterator = iter(train_loader)

    # Definir la función de pérdida y el optimizador
    criterian = nn.CrossEntropyLoss(ignore_index=target_to_index[PADDING_TOKEN],
                                    reduction='none')

    # When computing the loss, we are ignoring cases when the label is the padding token
    for params in transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    optim = torch.optim.Adam(transformer.parameters(), lr=lr)
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
                kn_predictions.view(-1, kn_vocab_size).to(device),
                labels.view(-1).to(device)
            ).to(device)
            valid_indicies = torch.where(labels.view(-1) == target_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            #train_losses.append(loss.item())

            #codigo agregado
            # Calcular BLEU y ROUGE para la predicción actual
            kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in kn_sentence_predicted:
                if idx == target_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_target[idx.item()]

            reference_sentence = kn_batch[0]
            bleu_score = sentence_bleu([reference_sentence.split()], predicted_sentence.split(),smoothing_function=smoothing )
            total_bleu_score += bleu_score

            rouge_scores = scorer.score(reference_sentence, predicted_sentence)
            for key in total_rouge_score:
                total_rouge_score[key] += rouge_scores[key].fmeasure


            #DUDA
            if batch_num % 1000 == 0:
                print(f"Iteration {batch_num} : {loss.item()}")
                print(f"{name_source_lang}: {eng_batch[0]}")
                print(f"{name_target_lang} Translation: {kn_batch[0]}")
                kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in kn_sentence_predicted:
                    if idx == target_to_index[END_TOKEN]:
                      break
                predicted_sentence += index_to_target[idx.item()]
                print(f"{name_target_lang} Prediction: {predicted_sentence}")


                #transformer.eval()
                #kn_sentence = ("",)
                #eng_sentence = ("Jesusqa chay runakunatam mikuchirqa pichqa tantallawan hinaspa iskay challwallawan.",)
                #eng_sentence = ("Los pollitos están piando por falta de comida.",)
                #for word_counter in range(max_sequence_length):
                #    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence)
                #    predictions = transformer(eng_sentence,
                #                              kn_sentence,
                #                              encoder_self_attention_mask.to(device),
                #                              decoder_self_attention_mask.to(device),
                #                              decoder_cross_attention_mask.to(device),
                #                              enc_start_token=False,
                #                              enc_end_token=False,
                #                              dec_start_token=True,
                #                              dec_end_token=False)
                #    next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                #    next_token_index = torch.argmax(next_token_prob_distribution).item()
                #    next_token = index_to_target[next_token_index]
                #    kn_sentence = (kn_sentence[0] + next_token, )
                #    if next_token == END_TOKEN:
                #      break

                #print(f"Evaluation translation (Los gatos  van a cazar gorriones por falta de comida.) : {kn_sentence}")
                #print("-------------------------------------------")

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
