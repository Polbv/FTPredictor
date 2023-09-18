# FTPredictor
A Basketeball FT percentage prediction model based on pose estimation and LSTM



  ![ezgif com-video-to-gif](https://github.com/Polbv/FTPredictor/assets/51133757/ff439561-6409-4aae-9f92-e99b9acce6ca)





## Installation

(python 3.8)
```
pip install torch torchvision torchaudio
pip install tensorboard
pip install opencv
pip install numpy
pip install matplotlib
```
follow mmpose installation: https://mmpose.readthedocs.io/en/latest/installation.html

## Usage
use inference.py to create a dataset containing inferenced videos and annotations in .json format (one annotation per video clip) and save it in /data.
Inspect and process the data using utils.py functions, use datasets.py to train LSTM models and dataset class. 
an example can be found in main.py


