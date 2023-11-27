import argparse
from vae_model import VariationalAutoencoder
from utils import plot_loss, save_digit_samples
from fastai.vision.all import *  # This imports most of the necessary vision-related modules
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.basic_train import Learner, DataBunch
from fastai.callback import EarlyStoppingCallback
from data_loader import load_data, check_loaded_data

data_path = 'data/even_mnist.csv'

# Load the data
train_dl, valid_dl = load_data(data_path)

# Check if the data is loaded correctly
check_loaded_data(train_dl)