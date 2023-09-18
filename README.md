# FTPredictor
A Basketeball FT percentage prediction model based on pose estimation and LSTM


https://github.com/Polbv/FTPredictor/assets/51133757/e1ac42f3-f464-4136-83ed-8b49c396598d





##Installation

python(3.8)
```
pip install torch torchvision torchaudio
pip install tensorboard
pip install opencv
pip install numpy
pip install matplotlib
```
follow mmpose installation: https://mmpose.readthedocs.io/en/latest/installation.html

##Usage
use inference.py to create a dataset containing inferenced videos and annotations in .json format (one annotation per video clip) ad save it in /data
inspect and process the data using utils functions, use datasets.py to train LSTM models and dataset class. 
an example can be found in main.py


