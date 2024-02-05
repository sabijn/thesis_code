from sklearn.metrics import f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import torch
import tqdm

from models import SpanProbe
    

def train_probe(train_data, dev_data, hidden_size, label_vocab, config):
    train_states, train_span_ids, train_labels = train_data
    dev_states, dev_span_ids, dev_labels = dev_data
    train_size = len(train_states)
    
    span_probe = SpanProbe(hidden_size, len(span_label_vocab), hidden_dropout_prob=0.).to(DEVICE)

    optimizer = optim.AdamW(span_probe.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_function = nn.CrossEntropyLoss()

    loss_curve = []
    train_accs = []
    dev_accs = []
    
    best_probe = None
    best_dev_f1 = 0.

    print(f"train f1\tdev f1\t\t--\tmerged f1\tmerged dev f1")
    
    try:
        for epoch in tqdm(range(config.epochs)):
            random_ids = np.array(random.sample(range(train_size), k=train_size))
            batch_size = 48

            batch_iterator = [
                [
                    (train_states[batch_idx], train_span_ids[batch_idx], train_labels[batch_idx])
                    for batch_idx in batch_ids
                ]
                for batch_ids 
                in np.array_split(random_ids, (train_size // config.batch_size - 1))
            ]

            for batch in batch_iterator:
                loss = 0.

                for batch_states, batch_span_ids, batch_labels in batch:
                    pred = span_probe(batch_states.unsqueeze(0), batch_span_ids)
                    loss += loss_function(pred, batch_labels)

                loss.backward()
                nn.utils.clip_grad_norm_(span_probe.parameters(), 0.25)
                optimizer.step()

                loss_curve.append(loss.detach().item())

            train_f1, train_merged_f1, _ = eval_probe(
                span_probe, 
                train_states, 
                train_span_ids, 
                train_labels,
                label_vocab,
            )
            dev_f1, dev_merged_f1, _ = eval_probe(
                span_probe, 
                dev_states, 
                dev_span_ids, 
                dev_labels,
                label_vocab,
            )
            train_accs.append(train_f1)
            dev_accs.append(dev_f1)

            if dev_merged_f1 > best_dev_f1:
                best_dev_f1 = dev_merged_f1
                best_probe = deepcopy(span_probe)

            if config.verbose:
                print(f"{train_f1:.3f} {dev_f1:.3f}\t\t--\t{train_merged_f1:.3f} {dev_merged_f1:.3f}")
    except KeyboardInterrupt:
        print(f"Interrupting training at epoch {epoch}")
        pass

    return best_probe, loss_curve, train_accs, dev_accs


def eval_probe(probe, states, spans, labels, label_vocab):
    probe.eval()
    
    all_labels = []
    all_preds = []
    
    for state, span, label in zip(states, spans, labels):
        with torch.no_grad():
            pred = probe(state.unsqueeze(0), span)
            pred = pred.argmax(-1)
            
            all_labels.extend(label.tolist())
            all_preds.extend(pred.tolist())
    
    probe.train()
    
    f1 = f1_score(all_labels, all_preds, average="micro")
    conf = confusion_matrix(all_labels, all_preds)

    base_labels = set(label.split("_")[0] for label in label_vocab.keys())
    idx_to_label = list(label_vocab.keys())

    correct = 0
    wrong = 0

    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            label_i = idx_to_label[i]
            label_j = idx_to_label[j]

            if label_i.split("_")[0] == label_j.split("_")[0]:
                correct += conf[i,j]
            else:
                wrong += conf[i,j]
    
    return f1, correct / (correct+wrong), (all_labels, all_preds)


def probe_loop(states, span_ids, labels, hidden_size, label_vocab, config):
    corpus_size = len(states)
    train_split, dev_split, test_split = int(0.8 * corpus_size), int(0.9 * corpus_size), corpus_size

    train_states = states[:train_split]
    dev_states = states[train_split:dev_split]
    test_states = states[dev_split:test_split]
    
    train_span_ids = span_ids[:train_split]
    dev_span_ids = span_ids[train_split:dev_split]
    test_span_ids = span_ids[dev_split:test_split]    
    
    train_labels = labels[:train_split]
    dev_labels = labels[train_split:dev_split]
    test_labels = labels[dev_split:test_split]

    span_probe, loss_curve, train_accs, dev_accs = train_probe(
        (train_states, train_span_ids, train_labels),
        (dev_states, dev_span_ids, dev_labels),
        hidden_size,
        label_vocab,
        config,
    )

    test_f1, test_merged_f1, test_preds = eval_probe(
        span_probe, 
        test_states, 
        test_span_ids, 
        test_labels,
        label_vocab,
    )
    
    if config.verbose:
        print((test_f1, test_merged_f1))
    
    return span_probe, loss_curve, train_accs, dev_accs, (test_f1, test_merged_f1), test_preds
