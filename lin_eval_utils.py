import re
from datasets import load_dataset
from transformers import EvalPrediction
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score


def get_dataset(dataset_name):
    if dataset_name == 'ecthr':
        return load_dataset('lex_glue', 'ecthr_b'), 10
    if dataset_name == 'scotus':
        return load_dataset('lex_glue', 'scotus'), 14
    if dataset_name == 'mimic':
        return load_dataset('kiddothe2b/multilabel_bench', name='mimic-l1'), 19
    raise ValueError('Invalid dataset {}'.format(dataset_name))


def get_preprocess_function(dataset_name):
    if dataset_name == 'ecthr':
        return preprocess_function_ecthr
    if dataset_name == 'scotus':
        return preprocess_function_scotus
    if dataset_name == 'mimic':
        return preprocess_function_mimic


def preprocess_function_scotus(examples, tokenizer, label_list):
    padding = 'max_length'
    max_seq_length = 4096
    cases = []
    for case in examples['text']:
        case = re.split('\n{2,}', case)
        cases.append(f' {tokenizer.sep_token} '.join([fact for fact in case]))
    batch = tokenizer(cases, padding=padding, max_length=max_seq_length, truncation=True)
    # use global attention on CLS token
    global_attention_mask = np.zeros((len(cases), max_seq_length), dtype=np.int32)
    global_attention_mask[:, 0] = 1
    batch['global_attention_mask'] = list(global_attention_mask)
    batch['label'] = [label_list.index(labels) for labels in examples['label']]
    return batch


def preprocess_function_ecthr(examples, tokenizer, label_list):
    padding = 'max_length'
    max_seq_length = 4096
    cases = []
    for case in examples['text']:
        cases.append(f' {tokenizer.sep_token} '.join([fact for fact in case]))
    batch = tokenizer(cases, padding=padding, max_length=max_seq_length, truncation=True)
    # use global attention on CLS token
    global_attention_mask = np.zeros((len(cases), max_seq_length), dtype=np.int32)
    global_attention_mask[:, 0] = 1
    batch['global_attention_mask'] = list(global_attention_mask)
    batch['labels'] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in examples['labels']]
    return batch


def preprocess_function_mimic(examples, tokenizer, label_list):
    padding = 'max_length'
    max_seq_length = 4096
    cases = []
    for case in examples['text']:
        case = re.split('\n{2,}', case)
        cases.append(f' {tokenizer.sep_token} '.join([fact for fact in case]))
    batch = tokenizer(cases, padding=padding, max_length=max_seq_length, truncation=True)
    # use global attention on CLS token
    global_attention_mask = np.zeros((len(cases), max_seq_length), dtype=np.int32)
    global_attention_mask[:, 0] = 1
    batch['global_attention_mask'] = list(global_attention_mask)
    batch['labels'] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in examples['labels']]
    return batch


def get_compute_metrics(dataset_name):
    if dataset_name == 'ecthr':
        return compute_metrics_ecthr
    if dataset_name == 'scotus':
        return compute_metrics_scotus
    if dataset_name == 'mimic':
        return compute_metrics_mimic


def compute_metrics_ecthr(eval_pred: EvalPrediction):
    # Fix gold labels
    y_true = np.zeros((eval_pred.label_ids.shape[0], eval_pred.label_ids.shape[1] + 1), dtype=np.int32)
    y_true[:, :-1] = eval_pred.label_ids
    y_true[:, -1] = (np.sum(eval_pred.label_ids, axis=1) == 0).astype('int32')
    # Fix predictions
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    y_pred = np.zeros((eval_pred.label_ids.shape[0], eval_pred.label_ids.shape[1] + 1), dtype=np.int32)
    y_pred[:, :-1] = preds
    y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype('int32')
    # Compute scores
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}


def compute_metrics_scotus(eval_pred: EvalPrediction):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}


def compute_metrics_mimic(eval_pred: EvalPrediction):
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    label_ids = (eval_pred.label_ids > 0.5).astype('int32')
    macro_f1 = f1_score(y_true=label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}