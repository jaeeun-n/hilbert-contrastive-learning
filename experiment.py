import argparse

from train import train
from linear_evaluation import lin_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model and do linear evaluation')
    parser.add_argument('--dataset', type=str, help='Dataset to train on (ecthr, scotus, mimic)')
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--loss_name', type=str)

    parser.add_argument('--path_model', type=str)
    parser.add_argument('--path_train_output', type=str)
    parser.add_argument('--path_data', type=str)

    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--train_num_epochs', type=int)
    parser.add_argument('--steps_per_epoch', type=int)
    parser.add_argument('--train_learning_rate', type=float)

    parser.add_argument('--simcse_temperature', type=float)
    parser.add_argument('--hilbert_temperature', type=float)
    parser.add_argument('--lambda_value', type=float)
    parser.add_argument('--embedding_size', type=int)

    parser.add_argument('--path_eval_output', type=str)
    parser.add_argument('--eval_num_epochs', type=int)
    parser.add_argument('--eval_learning_rate', type=float)

    args = parser.parse_args()
    print(args)

    loss_params = {'simcse_temperature': args.simcse_temperature, 'hilbert_temperature': args.hilbert_temperature,
                   'lambda_value': args.lambda_value, 'embedding_size': args.embedding_size}

    model_output_path = train(
        dataset_name=args.dataset, data_type=args.data_type, loss_name=args.loss_name,
        path_model=args.path_model, path_output=args.path_train_output, path_data=args.path_data,
        train_batch_size=args.train_batch_size, num_epochs=args.train_num_epochs, steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.train_learning_rate,
        loss_params=loss_params
    )
    lin_eval(
        dataset_name=args.dataset, path_model=model_output_path, path_output=args.path_eval_output,
        num_epochs=args.eval_num_epochs, learning_rate=args.eval_learning_rate
    )
