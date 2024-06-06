# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
from transformers import T5Tokenizer, T5EncoderModel
from format_input import *
import h5py
import torch
import re
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

# Load ProtT5 model
transformer_link = "Rostlab/ProstT5"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)

if device == torch.device("cpu"):
    print("Casting model to full precision for running on CPU ...")
    model.to(torch.float32)  # only cast to full-precision if no GPU is available

model = model.to(device)
model = model.eval()

tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

# get the sequence dictionary
data = read_data("data/full_filtered.csv")
seq_dict = sequence_dict(data)
# load embedding file to skip already processed sequences
if os.path.exists("embeddings.h5"):
    with h5py.File("embeddings.h5", "r") as f:
        processed_sequences = list(f.keys())
        seq_dict = {k: v for k, v in seq_dict.items() if k not in processed_sequences}

# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
# formatted_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq_dict]
formatted_sequences = {k: " ".join(list(re.sub(r"[UZOB]", "X", v))) for k, v in seq_dict.items()}

formatted_sequences = {k: "<AA2fold>" + " " + v for k, v in formatted_sequences.items()}
print("Formatted sequences: {}".format(len(formatted_sequences)))

# create batches of 64 sequences
batch_size = 2
batches = create_batches(formatted_sequences, batch_size)

print("Number of batches: {} of length {}".format(len(batches), batch_size))

c = 0
skipped = 0
for batch in batches:
    # tokenize sequences and pad up to the longest sequence in the batch
    print(f"Formatted sequences, starting encoding of batch {c} ...")
    ids = tokenizer.batch_encode_plus(batch.values(), add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    print("Generating embeddings ...")
    with torch.no_grad():
        try:
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        except torch.cuda.OutOfMemoryError:
            print("Out of memory error, skipping batch ...")
            skipped += 1
            c += 1
            continue

    with h5py.File("embeddings.h5", "a") as f:
        batch_num = 1
        for i, key in enumerate(batch.keys()):
            sequence = seq_dict[key]
            print(f"({batch_num}/{batch_size}) PDBchain: {key}")
            # print("Sequence: {}".format(sequence))
            # print("Embedding shape: {}".format(embedding_repr.last_hidden_state[i, :len(sequence)].shape))
            tensor = embedding_repr.last_hidden_state[i, :len(sequence)]
            f.create_dataset(key, data=tensor.cpu().numpy())
            batch_num += 1
    c += 1

print("Skipped {} batches due to memory errors.".format(skipped))
