import tempfile
import time
from datetime import datetime
from pathlib import Path
import argparse
import os.path
import gzip

import ray
from ray import train, tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.train import RunConfig

import torch
from torch import cuda
from torch.utils.data import DataLoader
import math
import numpy as np
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,\
    EarlyStoppingCallback, Trainer, TrainingArguments
from sentence_transformers import models, SentenceTransformer, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device

import train_utils
import lin_eval_utils


def objective(config):
    loss_params = {'simcse_temperature': config['simcse_temperature'],
                   'hilbert_temperature': config['hilbert_temperature'],
                   'lambda_value': config['lambda_value'],
                   'embedding_size': config['embedding_size']}

    training(dataset_name=config['dataset_name'],
             data_type=config['data_type'],
             loss_name=config['loss_name'],
             path_data=config['path_data'],
             path_model=config['path_model'],
             path_output=config['path_output'],

             steps_per_epoch=config['steps_per_epoch'],

             train_batch_size=config['train_batch_size'],
             num_epochs=config['num_epochs'],
             learning_rate=config['learning_rate'],

             loss_params=loss_params
             )


class ClusteringEvaluator(SentenceEvaluator):
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", path_output: str = ""):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.path_output = path_output

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        path_model = self.path_output + '/tmp_model'
        model.save(path_model)

        dataset, num_labels = lin_eval_utils.get_dataset(self.name)
        label_list = list(range(num_labels))

        tokenizer = AutoTokenizer.from_pretrained(path_model)
        if self.name == 'scotus':
            model = AutoModelForSequenceClassification.from_pretrained(path_model, num_labels=num_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path_model, num_labels=num_labels,
                                                                       problem_type='multi_label_classification')

        # Freeze embeddings (layers apart the classification head)
        for param in model.longformer.parameters():
            param.requires_grad = False

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)  # fp16
        preprocess_function = lin_eval_utils.get_preprocess_function(self.name)
        tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=['text'],
                                     fn_kwargs={'tokenizer': tokenizer, 'label_list': label_list})
        compute_metrics = lin_eval_utils.get_compute_metrics(self.name)

        tokenized_data.set_format("torch")
        if self.name != 'scotus':
            tokenized_data = (tokenized_data
                              .map(lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
                              .rename_column("float_labels", "labels"))

        output_dir = self.path_output + '/tmp_lin'

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            fp16=True,
            do_eval=False,
            logging_strategy='no',
            load_best_model_at_end=True,
            per_device_eval_batch_size=16,
            eval_steps=500,
            save_strategy="steps",
            evaluation_strategy="steps",
            metric_for_best_model="micro-f1",
            greater_is_better=True,
            save_total_limit=1
        )

        cuda.empty_cache()
        hist = Trainer(
            model=model,
            compute_metrics=compute_metrics,
            args=training_args,
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

        hist.train()
        score = hist.evaluate(eval_dataset=tokenized_data['validation'])

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            train.report(
                {"micro_f1": score['eval_micro-f1']},
                checkpoint=train.Checkpoint.from_directory(checkpoint_dir))

        return score['eval_micro-f1']


def training(dataset_name, data_type, path_model, train_batch_size, loss_name, num_epochs, path_output, steps_per_epoch,
             learning_rate, path_data, loss_params):
    # Prepare training corpus
    train_samples = train_utils.get_train_samples(dataset_name=dataset_name, data_type=data_type, path_data=path_data)
    train_dataloader = DataLoader(dataset=train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

    # Prepare validation corpus
    dataset, num_labels = lin_eval_utils.get_dataset(dataset_name)
    preprocess_function = lin_eval_utils.get_preprocess_function(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(path_model)
    label_list = list(range(num_labels))
    tokenized_val_data = dataset['validation'].map(preprocess_function, batched=True, remove_columns=['text'],
                                                   fn_kwargs={'tokenizer': tokenizer, 'label_list': label_list})
    tokenized_val_data.set_format("torch")
    if dataset_name != 'scotus':
        tokenized_val_data = (tokenized_val_data
                              .map(lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
                              .rename_column("float_labels", "labels"))

    tokenized_val_dataloader = DataLoader(tokenized_val_data, batch_size=16)

    # Define model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    word_embedding_model = models.Transformer(path_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    # Set training
    train_loss, hyperparameters = train_utils.get_loss(
        loss_name=loss_name, model=model, device=device, loss_params=loss_params)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    evaluator = ClusteringEvaluator(dataloader=tokenized_val_dataloader, name=dataset_name, path_output=path_output)

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_output_path = '{}/final-{}'.format(path_output, dt)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=10000,
        evaluator=evaluator,
        steps_per_epoch=steps_per_epoch,
        output_path=model_output_path,
        optimizer_params={'lr': learning_rate},
        use_amp=True  # Set to True, if your GPU supports FP16 cores
    )
    return model_output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--dataset', type=str, help='Dataset to train on (ecthr, scotus, mimic)')
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--loss_name', type=str)

    parser.add_argument('--path_model', type=str)
    parser.add_argument('--path_output', type=str)
    parser.add_argument('--path_data', type=str)

    args = parser.parse_args()
    print(args)

    search_space = {
        "dataset_name": args.dataset,
        "data_type": args.data_type,
        "loss_name": args.loss_name,
        "path_model": args.path_model,
        "path_output": args.path_output,
        "path_data": args.path_data,

        "steps_per_epoch": 2500,

        "learning_rate": tune.choice([1e-7, 1e-6, 1e-5]),
        "train_batch_size": 2,
        "num_epochs": tune.choice([10, 15, 20]),

        "lambda_value": tune.choice([0.5, 1, 1.5, 2, 2.5, 3, 3.5]),
        "embedding_size": tune.choice([64, 128]),
        "simcse_temperature": 0.1,
        "hilbert_temperature": 5,
    }

    algo = TuneBOHB(metric='micro_f1', mode='max', seed=1234)
    scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=10000)

    num_samples = 50
    trainable_with_gpu = tune.with_resources(objective, {"gpu": 1})

    if not os.path.exists(args.path_output + '/tmp_model'):
        os.mkdir(args.path_output + '/tmp_model')
    if not os.path.exists(args.path_output + '/tmp_lin'):
        os.mkdir(args.path_output + '/tmp_lin')

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            search_alg=algo,
            metric="micro_f1",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler
        ),
        param_space=search_space,
        run_config=RunConfig(name='_'.join([args.dataset, args.data_type]))
    )
    results = tuner.fit()
