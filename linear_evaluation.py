from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,\
    EarlyStoppingCallback, Trainer, TrainingArguments

import argparse
import torch
from torch import cuda

from lin_eval_utils import get_dataset, get_preprocess_function, get_compute_metrics


def lin_eval(dataset_name, path_model, path_output, learning_rate, num_epochs):
    dataset, num_labels = get_dataset(dataset_name)
    label_list = list(range(num_labels))

    tokenizer = AutoTokenizer.from_pretrained(path_model)
    if dataset_name == 'scotus':
        model = AutoModelForSequenceClassification.from_pretrained(path_model, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path_model, num_labels=num_labels,
                                                                   problem_type='multi_label_classification')

    # Freeze embeddings (layers apart the classification head)
    for param in model.longformer.parameters():
        param.requires_grad = False

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)  # fp16
    preprocess_function = get_preprocess_function(dataset_name)
    tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=['text'],
                                 fn_kwargs={'tokenizer': tokenizer, 'label_list': label_list})
    compute_metrics = get_compute_metrics(dataset_name)

    tokenized_data.set_format("torch")
    if dataset_name != 'scotus':
        tokenized_data = (tokenized_data
                          .map(lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
                          .rename_column("float_labels", "labels"))

    output_dir = '{}/lin_eval-{}'.format(path_output, path_model.replace('/', '_'))

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        fp16=True,
        eval_steps=500,
        save_strategy="steps",
        evaluation_strategy="steps",
        metric_for_best_model="micro-f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1
    )

    cuda.empty_cache()
    hist = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        args=training_args,
        eval_dataset=tokenized_data['validation'],
        train_dataset=tokenized_data['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    hist.train()
    results = hist.evaluate(eval_dataset=tokenized_data['test'])
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--dataset', default='ecthr', type=str, help='Dataset to train on (ecthr, scotus, mimic)')

    parser.add_argument('--path_model', default='output/final')
    parser.add_argument('--path_output', default='output/linear_evaluation')

    parser.add_argument('--eval_num_epochs', default=20, type=int)
    parser.add_argument('--eval_learning_rate', default=3e-5, type=float)

    args = parser.parse_args()
    print(args)

    lin_eval(dataset_name=args.dataset, path_model=args.path_model, path_output=args.path_output,
             num_epochs=args.eval_num_epochs, learning_rate=args.eval_learning_rate)
