# main.py
import argparse
import json
from training import train_model

def main():
    parser = argparse.ArgumentParser(description='Trains an RNN to perform multiplication of binary integers A * B = C')
    parser.add_argument('--param', type=str, help='file containing hyperparameters', required=True)
    parser.add_argument('--train-size', type=int, help='size of the generated training set', required=True)
    parser.add_argument('--test-size', type=int, help='size of the generated test set', required=True)
    parser.add_argument('--seed', type=int, help='random seed used for creating the datasets', required=True)

    args = parser.parse_args()

    # Load parameters from the JSON file
    with open(args.param, 'r') as file:
        params = json.load(file)

    # Pass arguments and parameters to the training function
    train_model(args.train_size, args.test_size, args.seed, params)

if __name__ == "__main__":
    main()

#python main.py --param param/param.json --train-size 10000 --test-size 1000 --seed 1234
