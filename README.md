# Plate

from an image by haar cascade get plate,
then from plate get regions (by MSER) ,
from regions get letters/digits by SVM/logistic regression

# Usage:
     python3 image2characters.py "filename"

# NOTE:
	the path one above current directory (Plate) must be in $PYTHONPATH
	for neural network you need weight.npz

# Background:
# Haar cascade description:
https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
raw training data in mka@mka-HP:~/PycharmProjects/Rekkari/Training

# The SVM/logistic regression files are trained in
└── TrainSVM
    ├── Digits
    │   └── SvmDir
    ├── Letters
    │   └── SvmDir

copied as follows
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Letters/SvmDir/logistic.pkl letters_logistic.pkl
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Letters/SvmDir/allSVM.txt.dict letters_logreg.dict
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Digits//SvmDir/logistic.pkl digits_logistic.pkl
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Digits//SvmDir/allSVM.txt.dict  digits_logreg.dict

# Sphinx:
sphinx-quickstart
> autodoc: automatically insert docstrings from modules (y/n) [n]: y
edit index.rst
make html