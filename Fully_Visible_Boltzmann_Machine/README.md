Homework 2: Fully Visible Boltzmann Machine Training

### Note: It's recommended to run the program multiple times to observe variations in the KL divergence graph. This is particularly useful if the initial run results in suboptimal performance. Repeated trials can help in obtaining a more representative view of the model's behavior.

# Please refer to the output_results_HW2 file to review my results.

# Note: If you encounter issues with the location of in.txt, consider modifying the file path in line 36 of the code:

parser.add_argument('file_path', nargs='?', default='data\\in.txt', type=str, help='Path to the input data file')
Try using alternative path formats like 'data/in.txt' or r'data/in.txt'.

## Dependencies

Requirements

Python 3.8 or newer
NumPy 1.19.2 or newer
Matplotlib 3.3.2 or newer

_full requirements listed below, if needed_

## Description of Task and Solution

Task Overview

The primary goal of this project was to develop a computational model capable of learning from data generated by a 1-D classical Ising chain. This chain is described by a Hamiltonian function, which in turn depends on unknown coupler values (J\_{i, j}). The challenge was to predict these coupler values accurately using only the structure of the 1-D closed chain and a dataset of spin configurations generated from a Monte-Carlo simulation of the Ising model.

The Ising model, in this context, represents a simplified physical system where spins can have values of either +1 or -1. These spins interact with their nearest neighbors, and the nature and strength of these interactions (represented by the coupler values) determine the system's overall state.

# Solution Strategy

Data Preparation and Representation

Data Importing: The data, representing spin configurations, was imported from a provided text file. Each line in this file corresponds to a distinct configuration of the Ising model.
Data Conversion: The spin values, originally represented as '+' and '-' in the file, were converted to binary values (+1 and -1) to facilitate mathematical operations.
Model Development

Model Choice: A Fully Visible Boltzmann Machine (FVBM) was selected as the model for this task. The FVBM is particularly suited for this kind of problem because it can effectively model systems where all parts (spins, in this case) interact with each other.

Parameter Initialization: The model parameters, including the coupler values, were initialized. In the absence of prior knowledge about the coupler values, they were either set to ones or initialized randomly.

Training Process
Partition Function Calculation: A key step in training involved calculating the partition function, which is critical in understanding the statistical mechanics of the Ising model.

    Probability Distribution Computation: The model computed the probability distribution of each data configuration based on the current model parameters.
    Parameter Update: Used the Metropolis-Coupled Markov Chain (MCMH) algorithm the model parameters were updated iteratively to maximize the likelihood of the observed data.
    Output Generation
    Coupler Value Prediction: After training, the model outputted its best guess of the coupler values, providing a clear and interpretable result.
    Performance and Visualization

KL Divergence Tracking: The training process included the calculation and tracking of the Kullback–Leibler (KL) divergence. This metric provided insights into how well the model's predictions matched the distribution of the training data.

Visualization: A plot of the KL divergence over training epochs was generated, offering a visual representation of the model's learning progress and convergence behavior.
Challenges and Solutions

Computational Complexity: Given the potentially large number of interactions in the Ising model, managing computational complexity was a key challenge. Efficient data structures and algorithms were employed to handle this.

Model Stability: Ensuring the stability of the model (preventing parameter divergence or NaN values) was crucial. Techniques from statistical physics and machine learning were combined to address this challenge.

# How AI Helped:

I used chatGPT to generate some funcitons to make process fasted and copilot helped me to create comments. In this HW I also used GPT to understand the phycics of the process

# How to Run My Code

Here's how to make my code work:

Set-Up: First, make sure your computer has all the libs listed above installed and you are in HW2 folder.

Run the Script: Type this into your computer to run `main.py`, use:

bash

# python main.py data/in.txt

Help Command: If you're stuck or need more info, just type:

bash

# python main.py --help

This command will start the training process using the provided input data file.

Input Data Format
The input data should be in a text file (e.g., in.txt) representing the spin configurations of the Ising model. Each line in the file corresponds to a spin configuration in the format s*0 s_1 s_2 ... s*{N-1}, where N is the size of the model.

Output
The program outputs its best guess of the correct values of all J\_{i, j} in the 1-D Ising chain from which the training dataset was generated. The output is a dictionary of couplers, with keys as pairs of indices and values as the predicted coupler strengths.

## Expected result:

Coupling of Generated data set: [-1. 1. 1. 1.]
Coupling of Original data set: [-1 1 1 1]
True

## Hyperparameters

You can configure hyperparameters to customize the training process. Hyperparameters include batch_size, num_epochs, and lr (learning rate).

To specify hyperparameters, you have two options:

# Option 1: Command Line Arguments

You can directly pass hyperparameters as command line arguments when running the script. For example:

# shell

# python main.py data/in.txt --batch_size 128 --num_epochs 200 --lr 0.01

# Option 2: JSON Parameter File

Alternatively, you can create a JSON parameter file to specify hyperparameters. Here's how to do it:

Create a JSON file, e.g., param.json, in the project directory.

Define the hyperparameters in the JSON file. For example:

json

{
"batch_size": 128,
"num_epochs": 200,
"lr": 0.01
}

When running the script, use the --param_file argument to specify the JSON parameter file:

shell

python main.py data/in.txt --param_file param.json

# Help

You will find this options to look for

--batch_size: Set the batch size for training.
--num_epochs: Specify the number of training epochs.
--lr: Adjust the learning rate for training.

# Requirements:

absl-py==1.4.0
aiohttp==3.8.4
aiosignal==1.3.1
asttokens==2.2.1
async-timeout==4.0.2
attrs==23.1.0
backcall==0.2.0
blis==0.7.9
cachetools==5.3.1
catalogue==2.0.8
certifi==2022.12.7
charset-normalizer==3.1.0
click==8.1.4
colorama==0.4.6
comm==0.1.3
confection==0.1.0
contourpy==1.0.7
cycler==0.11.0
cymem==2.0.7
debugpy==1.6.7
decorator==5.1.1
dill==0.3.7
distlib==0.3.6
executing==1.2.0
fastai==2.7.12
fastcore==1.5.29
fastdownload==0.0.7
fastprogress==1.0.3
filelock==3.10.0
flatbuffers==23.3.3
fonttools==4.39.2
frozenlist==1.3.3
google-auth==2.22.0
google-auth-oauthlib==1.0.0
grpcio==1.57.0
h5py==3.8.0
idna==3.4
ipykernel==6.23.1
ipython==8.14.0
jedi==0.18.2
Jinja2==3.1.2
joblib==1.2.0
jupyter_client==8.2.0
jupyter_core==5.3.0
kiwisolver==1.4.4
langcodes==3.3.0
libclang==16.0.0
Markdown==3.4.4
MarkupSafe==2.1.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
mpmath==1.3.0
multidict==6.0.4
murmurhash==1.0.9
nest-asyncio==1.5.6
networkx==3.1
numpy==1.24.2
oauthlib==3.2.2
openai==0.27.7
opencv-python==4.7.0.72
packaging==23.0
pandas==2.0.3
parso==0.8.3
pathy==0.10.2
pbr==5.11.1
pickleshare==0.7.5
Pillow==9.4.0
platformdirs==3.1.1
ply==3.11
preshed==3.0.8
prompt-toolkit==3.0.38
protobuf==4.24.0
psutil==5.9.5
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.3.0
pydantic==1.10.11
Pygments==2.15.1
pylatexenc==2.10
pyparsing==3.0.9
PyPDF2==3.0.1
python-dateutil==2.8.2
python-version==0.0.2
pytz==2023.3
pywin32==306
PyYAML==6.0
pyzmq==25.1.0
qiskit==0.44.2
qiskit-terra==0.25.2.1
regex==2023.5.5
requests==2.28.2
requests-oauthlib==1.3.1
rsa==4.9
rustworkx==0.13.2
scikit-learn==1.2.2
scipy==1.10.1
seaborn==0.12.2
six==1.16.0
smart-open==6.3.0
spacy==3.6.0
spacy-legacy==3.0.12
spacy-loggers==1.0.4
srsly==2.4.6
stack-data==0.6.2
stevedore==5.1.0
sympy==1.11.1
tensorboard==2.14.0
tensorboard-data-server==0.7.1
tensorboard-plugin-wit==1.8.1
thinc==8.1.10
threadpoolctl==3.1.0
torch==2.0.0+cu117
torch-tb-profiler==0.4.1
torchaudio==2.0.1+cu117
torchinfo==1.8.0
torchsummary==1.5.1
torchvision==0.15.1
tornado==6.3.2
tqdm==4.65.0
traitlets==5.9.0
typer==0.9.0
typing_extensions==4.5.0
tzdata==2023.3
urllib3==1.26.15
virtualenv==20.21.0
virtualenvwrapper-win==1.2.7
wasabi==1.1.2
wcwidth==0.2.6
Werkzeug==2.3.7
wrapt==1.14.1
yarl==1.9.2