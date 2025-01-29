#!/bin/sh

# DO NOT put the installation of pytorch in this file
# Users need to check the following website to install pytorch
# https://pytorch.org/get-started/locally/
# The reason is that users can have different hardware settings

pip install --upgrade numpy~=1.26.4
pip install --upgrade scipy~=1.13.1
pip install --upgrade matplotlib~=3.10.0
pip install --upgrade ipython~=8.31.0
pip install --upgrade pandas~=2.2.3
pip install --upgrade plotly~=5.18.0
pip install --upgrade scikit-learn~=1.6.1
pip install --upgrade seaborn~=0.13.2
pip install --upgrade pytz~=2024.1
pip install --upgrade tqdm~=4.67.1
pip install --upgrade pyarrow~=15.0.2