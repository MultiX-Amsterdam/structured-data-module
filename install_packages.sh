#!/bin/sh

# DO NOT put the installation of pytorch in this file
# Users need to check the following website to install pytorch
# https://pytorch.org/get-started/locally/
# The reason is that users can have different hardware settings

pip install --upgrade numpy~=1.24
pip install --upgrade scipy~=1.12
pip install --upgrade matplotlib~=3.8
pip install --upgrade ipython~=8.7
pip install --upgrade pandas~=2.2
pip install --upgrade plotly~=5.18
pip install --upgrade scikit-learn~=1.4
pip install --upgrade seaborn~=0.13
pip install --upgrade pytz~=2024.1
pip install --upgrade tqdm~=4.66
pip install --upgrade pyarrow~=15.0.0
