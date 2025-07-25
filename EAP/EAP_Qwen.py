import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sys
from typing import List, Union, Optional, Tuple, Literal
from functools import partial
from IPython.display import Image, display

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache
import plotly.io as pio

pio.renderers.default = "colab"

if not torch.cuda.is_available():
    print("WARNING: Running on CPU. Did you remember to set your Colab accelerator to GPU?")


def setup_model():
    """Load and configure the pretrained Qwen1.5-1.8B model for EAP analysis"""
    model = HookedTransformer.from_pretrained("qwen-1.8b", device="cuda:0")  # Correct name for pretrained model
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.to(model.cfg.device)
    return model



def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    the probability difference metric, which takes in logits and labels (years), and
    returns the difference in prob. assigned to valid (> year) and invalid (<= year) tokens

    (corrupted_logits and input_lengths are due to the Graph framework introduced below)

    """
    probs = torch.softmax(logits[:, -1], dim=-1)
    results = []
    probs, next_tokens = torch.topk(probs[-1], 5)
    prob_a = 0
    prob_b = 0
    for prob, token, label in zip(probs, next_tokens, labels):
        if token == label:
            prob_b = prob
        else:
            prob_a = prob_a + prob

    results = prob_b - prob_a
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results

from sklearn.preprocessing import LabelEncoder

# def batch_dataset(df, batch_size=2):
#     clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]

#     # Convert string labels ('A', 'B', etc.) to numeric using LabelEncoder
#     le = LabelEncoder()
#     label = le.fit_transform(label)

#     # Store label encoder for inverse mapping later, if needed
#     df['encoded_label'] = label

#     # Batch the data
#     clean = [clean[i:i + batch_size] for i in range(0, len(df), batch_size)]
#     corrupted = [corrupted[i:i + batch_size] for i in range(0, len(df), batch_size)]
#     label = [torch.tensor(label[i:i + batch_size]) for i in range(0, len(df), batch_size)]

#     return [(clean[i], corrupted[i], label[i]) for i in range(len(clean))]


def batch_dataset(model, df, batch_size=1):
    from sklearn.preprocessing import LabelEncoder

    # Encode labels
    le = LabelEncoder()
    df['label'] = df['label'].astype(str)
    df['encoded_label'] = le.fit_transform(df['label'])

    # Prepare rows
    clean = df['clean'].tolist()
    corrupted = df['corrupted'].tolist()
    labels = df['encoded_label'].tolist()

    # Tokenize and group by length
    tokenized_lengths = [len(model.tokenizer(c).input_ids) for c in clean]
    length_buckets = {}
    for i, l in enumerate(tokenized_lengths):
        length_buckets.setdefault(l, []).append(i)
    print(length_buckets) 
    # Batch within each bucket
    batches = []
    for indices in length_buckets.values():
        for i in range(0, len(indices), batch_size):
            idxs = indices[i:i + batch_size]
            if len(idxs) < batch_size:
                continue  # Skip incomplete batch
            batch_clean = [clean[j] for j in idxs]
            batch_corrupted = [corrupted[j] for j in idxs]
            batch_label = torch.tensor([labels[j] for j in idxs])
            batches.append((batch_clean, batch_corrupted, batch_label))
    return batches



def validate_dataset_tokenization(model, dataset):
    """Ensure clean and corrupted examples have same tokenized length by padding/truncating corrupted as needed."""
    mask_token_id = model.tokenizer.convert_tokens_to_ids('<mask>') if '<mask>' in model.tokenizer.get_vocab() else model.tokenizer.eos_token_id
    fixed_dataset = []
    for clean, corrupted, label in dataset:
        clean_toks = model.tokenizer(clean).input_ids
        corrupted_toks = model.tokenizer(corrupted).input_ids
        new_corrupted = []
        for clean_example_toks, corrupted_example_toks in zip(clean_toks, corrupted_toks):
            len_clean = len(clean_example_toks)
            len_corr = len(corrupted_example_toks)
            if len_corr < len_clean:
                # Pad corrupted with <mask> token id
                corrupted_example_toks = corrupted_example_toks + [mask_token_id] * (len_clean - len_corr)
            elif len_corr > len_clean:
                # Truncate corrupted to match clean
                corrupted_example_toks = corrupted_example_toks[:len_clean]
            # Now, re-decode corrupted tokens to text
            new_corrupted.append(model.tokenizer.decode(corrupted_example_toks, skip_special_tokens=False))
        # Join if multiple sentences
        fixed_corrupted = new_corrupted[0] if len(new_corrupted) == 1 else ' '.join(new_corrupted)
        fixed_dataset.append((clean, fixed_corrupted, label))
    print("Dataset tokenization validation: all pairs aligned.")
    return fixed_dataset



import eap
from eap.graph import Graph
from eap import evaluate
from eap import attribute_mem as attribute

def get_important_edges(model, dataset, metric, top_k=400):
    """Run EAP and return the important edges"""
    # Validate dataset first
    validate_dataset_tokenization(model, dataset)

    # Create graph from model
    g = Graph.from_model(model)

    # Evaluate baseline
    baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    print(f"Baseline: {baseline}")

    # Run attribution
    attribute.attribute(model, g, dataset, partial(metric, loss=True, mean=True))

    # Apply threshold to get top edges
    scores = g.scores(absolute=True)
    #g.apply_threshold(scores[-top_k], absolute=True)
    # print("edge num after patching:",sum(edge.in_graph for edge in g.edges.values()))

    # Get edge information
    edges = {edge_id: {'score': edge.score, 'abs_score': abs(edge.score),
                       'source': str(edge.parent), 'target': str(edge.child)}
             for edge_id, edge in g.edges.items() if edge.in_graph}

    return g, edges

def compute_edge_overlap(edges1: dict, edges2: dict):
    """Compute overlap between two sets of edges"""
    # Get edge IDs from both sets
    edge_ids1 = set(edges1.keys())
    edge_ids2 = set(edges2.keys())

    # Compute intersection
    common_edges = edge_ids1.intersection(edge_ids2)

    # Overlap metrics
    overlap_count = len(common_edges)

    # Get details of common edges
    common_edge_details = []
    for edge_id in common_edges:
        common_edge_details.append({
            'edge_id': edge_id,
            'score1': edges1[edge_id]['score'],
            'score2': edges2[edge_id]['score'],
            'abs_score1': edges1[edge_id]['abs_score'],
            'abs_score2': edges2[edge_id]['abs_score'],
            'source': edges1[edge_id]['source'],
            'target': edges1[edge_id]['target']
        })

    # Sort by average absolute score
    common_edge_details.sort(key=lambda x: (x['abs_score1'] + x['abs_score2']) / 2, reverse=True)

    return {
        'overlap_count': overlap_count,
        'common_edges': common_edge_details
    }

def main():
    try:
        print(sys.argv[1])
        df1 = pd.read_csv("part/"+sys.argv[1]+".csv",sep="\t",on_bad_lines='skip',index_col=False)  # adjust this path if needed
        print(f"Loaded dataset with {len(df1)} samples")
        print(df1.columns)
    except FileNotFoundError:
        print("Error: 'a.csv' not found!")
        return

    # Load model first so tokenization is consistent
    try:
        model = setup_model()
        torch.cuda.empty_cache()
        print(f"Pretrained Qwen model loaded on {model.cfg.device}")
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        return

    # Batch dataset (model is now available)
    dataset1 = batch_dataset(model, df1, batch_size=1)
    print(f"Prepared {len(dataset1)} batches")

    if len(dataset1) == 0:
        print("❌ No batches prepared. Try reducing batch_size or inspect token lengths.")
        return

    # Define metric
    metric = prob_diff

    # Run EAP
    print("Running EAP on Qwen model...")
    g, edges = get_important_edges(model, dataset1, metric, top_k=400)


    

    pd.DataFrame.from_dict(edges, orient='index').to_csv('imp_edges_principle10/qwen_important_edges'+sys.argv[1]+".csv", index=True)
    print("Important edges saved to 'qwen_important_edges.csv'.")

    return g, edges


if __name__ == "__main__":
    main()
