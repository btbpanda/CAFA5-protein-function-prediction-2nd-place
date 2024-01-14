import argparse
import os
import re

import numpy as np
import torch
import tqdm
import yaml
from Bio import SeqIO
from transformers import EsmModel, EsmTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str, default="1")

model_names_sizes = [
    ("esm2_t33_650M_UR50D", 1280)
]

model_name = f'facebook/{model_names_sizes[0][0]}'


def get_embeddings(seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)

    return emb_0_per_protein


if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_cross_attention=False, is_decoder=False).to(device)
    model.eval()

    model_name = 'esm2_t33_650M_UR50D'
    kaggle_dataset = config['base_path']  # sys.argv[1]
    output_path = os.path.join(config['base_path'], config['embeds_path'], 'esm_small')  # sys.argv[2]
    os.makedirs(output_path, exist_ok=True)

    fn = os.path.join(kaggle_dataset, 'Train', 'train_sequences.fasta')
    sequences = SeqIO.parse(fn, "fasta")
    num_sequences = sum(1 for seq in sequences)

    ids = []
    embeds = np.zeros((num_sequences, 1280))
    i = 0
    for seq in tqdm.tqdm(sequences):
        ids.append(seq.id)
        embeds[i] = get_embeddings(str(seq.seq)).detach().cpu().numpy()
        i += 1
        # break

    np.save(os.path.join(output_path, f'train_embeds.npy'), embeds)
    np.save(os.path.join(output_path, f'train_ids.npy'), np.array(ids))

    fn = os.path.join(kaggle_dataset, 'Test (Targets)', 'testsuperset.fasta')

    sequences = SeqIO.parse(fn, "fasta")
    num_sequences = sum(1 for seq in sequences)
    print("Number of sequences in test:", num_sequences)
    sequences = SeqIO.parse(fn, "fasta")

    ids = []
    embeds = np.zeros((num_sequences, 1280))
    i = 0
    for seq in tqdm.tqdm(sequences):
        ids.append(seq.id)
        embeds[i] = get_embeddings(str(seq.seq)).detach().cpu().numpy()
        i += 1
        # break
    np.save(os.path.join(output_path, f'test_embeds.npy'), embeds)
    np.save(os.path.join(output_path, f'test_ids.npy'), np.array(ids))
