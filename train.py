import os.path
import re
import time
import torch
from torch.utils.data import DataLoader
import math
from datetime import datetime
import gzip
import tqdm
import argparse

from sentence_transformers import models
from sentence_transformers import SentenceTransformer, InputExample

import train_utils


def train(dataset_name, data_type, path_model, train_batch_size, loss_name, num_epochs, path_output, steps_per_epoch,
          learning_rate, path_data, loss_params):
    # Prepare training corpus
    train_samples = train_utils.get_train_samples(dataset_name=dataset_name, data_type=data_type, path_data=path_data)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    word_embedding_model = models.Transformer(path_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    print('Model used: ', model)
    print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

    train_dataloader = DataLoader(dataset=train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
    train_loss, hyperparameters = train_utils.get_loss(loss_name=loss_name, model=model, device=device,
                                                       loss_params=loss_params)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

    # Set output paths
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_output_path = '{}/final-{}'.format(path_output, dt)

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        steps_per_epoch=steps_per_epoch,
        output_path=model_output_path,
        optimizer_params={'lr': learning_rate},
        save_best_model=True,
        use_amp=True  # Set to True, if your GPU supports FP16 cores
    )
    t1 = time.time() - t0
    print("Time elapsed in hours: ", t1 / 3600)
    print('Final model: ', model_output_path)
    return model_output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset', type=str, help='Dataset to train on (ecthr, scotus, mimic)')
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--loss_name', type=str, help='Either ')

    parser.add_argument('--path_model', type=str)
    parser.add_argument('--path_output', type=str)
    parser.add_argument('--path_data', type=str)

    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--steps_per_epoch', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--embedding_size', type=int)

    parser.add_argument('--simcse_temperature', type=float)
    parser.add_argument('--hilbert_temperature', type=float)
    parser.add_argument('--lambda_value', type=float)
    parser.add_argument('--embedding_size', type=int)

    args = parser.parse_args()
    print(args)

    loss_params = {'simcse_temperature': args.simcse_temperature, 'hilbert_temperature': args.hilbert_temperature,
                   'lambda_value': args.lambda_value, 'embedding_size': args.embedding_size}

    train(
        dataset_name=args.dataset, loss_name=args.loss_name, path_model=args.path_model, path_output=args.path_output,
        train_batch_size=args.train_batch_size, num_epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate, data_type=args.data_type, path_data=args.path_data,
        loss_params=loss_params
    )