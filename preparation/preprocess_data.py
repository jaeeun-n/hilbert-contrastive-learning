import os.path
import re
import time
from typing import List

from datasets import load_dataset
import tqdm
import string

from sentence_transformers import models, InputExample
from sentence_transformers import SentenceTransformer, InputExample

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random


def preprocess_data(dataset_name, path_output, do_doc, do_par):
    if dataset_name == 'ecthr':
        preprocess_ecthr(path_output, do_doc, do_par)
        print('ECtHR preprocessed.')
    if dataset_name == 'scotus':
        preprocess_scotus_mimic(dataset_name, path_output, do_doc, do_par)
        print('SCOTUS preprocessd.')
    if dataset_name == 'mimic':
        preprocess_scotus_mimic(dataset_name, path_output, do_doc, do_par)
        print('MIMIC preprocessed.')
    raise ValueError('Invalid dataset {}'.format(dataset_name))


def preprocess_ecthr(path_output, do_doc, do_par):
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    doc_list = load_dataset(path='lex_glue', name='ecthr_b', split='train')

    if do_doc:
        path_output_file = path_output + '/ecthr_doc_doc'

        with open(path_output_file, 'w') as f:
            for example in doc_list['text']:
                example = ' '.join(example)
                f.write(f"{[example]}\n")

    if do_par:
        path_output_file_par = path_output + '/ecthr_par_par'
        path_output_file_par_idf_random = path_output + '/ecthr_par_par_idf_random'

        with (open(path_output_file_par, 'w') as fpar,
              open(path_output_file_par_idf_random, 'w') as frand):

            for doc in tqdm.tqdm(doc_list, desc='Generate paragraphs and sentences'):
                doc = doc['text']

                for par in doc:
                    par = re.sub('\n{2,}', '\n', par)
                    fpar.write(f"{[par]}\n")

                # idf weighted paragraphs
                idf_weight_per_par = get_idf_weights(doc)
                # filter examples with lowest idf
                n_lowest = int(0.1 * len(doc))
                indices = np.argpartition(idf_weight_per_par, n_lowest)
                idf_paragraphs = [doc[i] for i in indices[n_lowest:]]
                # randomly choose one of them
                sampled_paragraph = idf_paragraphs[random.randint(0, len(idf_paragraphs) - 1)]
                frand.write(f"{[sampled_paragraph]}\n")


def preprocess_scotus_mimic(dataset_name, path_output, do_doc, do_par):
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    if dataset_name == 'scotus':
        doc_string = load_dataset(
            path='lex_glue', name='scotus', split='train')
    else:
        doc_string = load_dataset(
            path='kiddothe2b/multilabel_bench', name='mimic-l1', split='train')

    if do_doc:
        path_output_file = path_output + '/' + dataset_name + '_doc_doc'

        with open(path_output_file, 'w') as f:
            for example in doc_string['text']:
                document_list = re.split('\n{2,}', example)
                f.write(f"{document_list}\n")

    if do_par:
        path_output_file_par = path_output + '/' + dataset_name + '_par_par'
        path_output_file_par_idf_random = path_output + '/' + dataset_name + '_par_par_idf_random'

        doc_list = doc_string.map(split_paragraphs)

        with (open(path_output_file_par, 'w') as fpar,
              open(path_output_file_par_idf_random, 'w') as frand):

            for doc in tqdm.tqdm(doc_list, desc='Generate paragraphs and sentences'):
                doc = doc['paragraphs']

                for par in doc:
                    fpar.write(f"{[par]}\n")

                # idf weighted paragraphs
                idf_weight_per_par = get_idf_weights(doc)
                # filter examples with lowest idf
                n_lowest = int(0.1 * len(doc))
                indices = np.argpartition(idf_weight_per_par, n_lowest)
                idf_paragraphs = [doc[i] for i in indices[n_lowest:]]

                # randomly choose one of them
                sampled_paragraph = idf_paragraphs[random.randint(0, len(idf_paragraphs) - 1)]
                frand.write(f"{[sampled_paragraph]}\n")


def get_idf_weights(document):
    count_model = CountVectorizer()
    count_matrix = count_model.fit_transform(document)
    count_array = count_matrix.toarray()

    n_sequence_with_word = np.sum(count_array, axis=0)
    n_sequences = len(document)

    idf = np.multiply(count_array, n_sequence_with_word[np.newaxis, :])
    idf = np.where(idf == 0, np.inf, idf)
    idf = n_sequences / idf

    mean_per_sequence = []
    for i in range(idf.shape[0]):
        row_sum = np.sum(idf[i])
        n_words = np.sum(count_array[i])
        mean_idf = row_sum / (n_words + 1e-10)  # Add a small value to avoid division by zero
        mean_per_sequence.append(mean_idf)

    return mean_per_sequence


def split_paragraphs(example):
    paragraphs = [paragraph for paragraph in re.split(r'\s*\n{2,}', example['text']) if len(paragraph) > 2]

    # Merge small paragraphs
    merged_paragraphs = []
    merged_paragraph = ''
    for idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if idx == 0:
            continue
        if len(paragraph.split()) < 32 and not re.match('[a-zA-Z ]{,50}:', paragraph[0]):
            merged_paragraph += ' ' + paragraph
        else:
            if len(merged_paragraph) > 0 and len(merged_paragraph.split()) < 32:
                merged_paragraphs.append(merged_paragraph + ' ' + paragraph)
            elif len(merged_paragraph) > 0:
                merged_paragraphs.append(merged_paragraph)
                merged_paragraphs.append(paragraph)
            else:
                merged_paragraphs.append(paragraph)
            merged_paragraph = ''

    if len(merged_paragraphs) == 0:
        merged_paragraphs = [example['text']]

    example['paragraphs'] = merged_paragraphs
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Training Data.')
    parser.add_argument('--dataset', type=str, help='Dataset to train on (ecthr, scotus, mimic)')
    parser.add_argument('--do_doc', type=bool, default=True, help='Whether create document training data')
    parser.add_argument('--do_par', type=bool, default=True, help='Whether create paragraph training data')
    parser.add_argument('--path_output', type=str, default='preprocessed_data')

    args = parser.parse_args()
    print(args)

    preprocess_data(dataset_name=args.dataset, do_doc=args.do_doc, do_par=args.do_par, path_output=args.path_output)
