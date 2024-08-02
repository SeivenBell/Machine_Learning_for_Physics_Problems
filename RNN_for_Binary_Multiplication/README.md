TO DO: create doc strings for functions

Work 1: RNN for Binary Multiplication
What You Need to Run This
You'll need these libraries to get my project up and running:

Python 3.8 or newer
PyTorch 1.7.1
NumPy 1.19.2
Matplotlib 3.3.2
argparse
json
time
random 

*full requirements below if needed*

Description of Task and Solution

Task Overview

The core task of this assignment was to develop a machine learning model, specifically a Recurrent Neural Network (RNN), capable of multiplying two integers. These integers are not in their usual decimal form but in binary representation. Each of the numbers, A_i and B_i, consists of 8 binary digits, and their product, C_i, extends to 16 binary digits. A unique aspect is that the numbers are stored in 'little endian' format, meaning the least significant bit is the first in the sequence, which is the reverse of the conventional order.

My Approach:

Recognizing the sequential nature of binary numbers and the repetitive process involved in multiplication, I chose an RNN for this task. RNNs excel in handling sequences, making them ideal for this kind of problem where the relation between sequential bits influences the output.

Solution Highlights:

Data Preparation: I used Python to manipulate and prepare the binary data. This involved converting binary strings into a format suitable for the RNN.

RNN Model: I built an RNN model capable of processing sequences of binary digits. The model takes interleaved bits from A_i and B_i (i.e., a_0 b_0 a_1 b_1 ... a_7 b_7) and learns to predict the resulting product bits (c_0 c_1 ... c_15).

Training and Testing: I trained the model using a generated dataset of binary multiplication examples. To evaluate the model's performance, I used both a training set and a separate test set, noting the model's ability to generalize to new, unseen data.

Swap Learning and Testing: Given the commutative nature of multiplication, I also tested the model's performance when the inputs A and B are swapped, assessing its understanding of the underlying mathematical principle rather than just memorizing patterns.

Performance Metrics: During training, I tracked and reported the model's loss (error rate), which provided insights into its learning progress. The final performance was gauged based on how accurately the model could predict the product of two binary numbers.

How AI Helped:

I used chatGPT to generate some funcitons to make process fasted and copilot helped me to create comments.

How to Run My Code
Here's how to make my code work:

Set-Up: First, make sure your computer has all the stuff listed above installed and you are in HW1 folder.
Run the Script: Type this into your computer to run `main.py`, use:

sh

python main.py --param param/param.json --train-size 10000 --test-size 1000 --seed 1337


Help Command: If you're stuck or need more info, just type:

sh

python main.py --help


Usage Documentation for the Configuration File

This document provides clear documentation for using the provided configuration file, which is structured as follows:

json

{
    "model": {
        "hidden_dim": 256
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 10
    }
}

The configuration file contains settings for configuring a machine learning model and its training process. Below, we explain the purpose and usage of each section and its parameters.

Model Configuration
The "model" section contains parameters related to the architecture and configuration of your machine learning model.

hidden_dim (integer)

Purpose: This parameter sets the number of hidden dimensions in the model. It determines the size of the hidden layers in your neural network model.
Usage: You can adjust this value according to your specific model requirements. Larger values may allow the model to capture more complex patterns but may also increase the computational cost.

Training Configuration
The "training" section contains parameters that control the training process of your machine learning model.

learning_rate (float)

Purpose: This parameter sets the learning rate for the optimizer. The learning rate controls the step size taken during gradient descent and affects how quickly your model converges during training.
Usage: Adjust this value based on your specific task. A higher learning rate may lead to faster initial training but could result in instability or overshooting optimal solutions. Conversely, a lower learning rate may lead to more stable training but require more epochs for convergence.


batch_size (integer)

Purpose: This parameter determines the number of data samples used in each iteration (batch) during training. It affects the trade-off between training speed and memory usage.
Usage: Choose an appropriate batch size based on your available GPU memory and dataset size. Smaller batch sizes consume less memory but may slow down training, while larger batch sizes can speed up training but require more memory.


num_epochs (integer)

Purpose: This parameter sets the number of times your model will iterate over the entire training dataset during training. It controls how many times your model learns from the entire dataset.
Usage: Set this value based on your specific task and dataset. Training for more epochs allows the model to learn more from the data but may risk overfitting if not properly controlled.

Requirements:

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
