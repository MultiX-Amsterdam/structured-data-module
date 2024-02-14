#!/usr/bin/env python
# coding: utf-8

# # PyTorch Implementation (Structured Data Processing)

# (Last updated: Feb 12, 2024)[^credit]
# 
# [^credit]: Credit: this teaching material is created by [Yen-Chia Hsu](https://github.com/yenchiah).

# In this practice, we will guide you using the PyTorch deep learning framework to implement a deep regression model on the Smell Pittsburgh dataset. We only provide the basics in this notebook, and the following resources give more detailed information about PyTorch:
# - [Introduction to PyTorch (Part 1), UvA Deep Learning Course](https://www.youtube.com/watch?v=wnKZZgFQY-E)
# - [Introduction to PyTorch (Part 2), UvA Deep Learning Course](https://www.youtube.com/watch?v=schbjeU5X2g)
# - [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
# 
# :::{important}
# To make this notebook work, you need to [install PyTorch](https://pytorch.org/get-started/locally/). You can also copy this notebook (as well as the dataset) to Google Colab and run the notebook on it.
# :::
# 
# First, we begin importing the required packages.

# In[1]:


import pandas as pd
import numpy as np
from os.path import isfile
from os.path import join
from os import listdir
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# Then, we need to set the resources that we want to use for computing. On a machine with a GPU (Graphics Processing Unit), we would prefer to use it since a GPU can speed up the computation a lot. On a MacOS computer with advanced chips (e.g., M1 or M2), PyTorch can use the [MPS backend](https://pytorch.org/docs/stable/notes/mps.html) to perform computing.

# In[2]:


if torch.cuda.is_available():
    device = torch.device("cuda") # use CUDA device
elif torch.backends.mps.is_available():
    device = torch.device("mps") # use MacOS GPU device (e.g., for M2 chips)
else:
    device = torch.device("cpu") # use CPU device
device


# The following code will help you move the data to the device that you choose.

# In[3]:


def to_device(data, device):
    """Move PyTorch objects (e.g., tensors, models) to a chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Below we hide a bunch of functions for preprocessing the data. These functions are from the structured data processing tutorial.

# In[4]:


def answer_preprocess_sensor(df_list):
    """
    This function is the answer of task 5.
    Preprocess sensor data.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        A list of data frames that contain sensor data from multiple stations.

    Returns
    -------
    pandas.DataFrame
        The preprocessed sensor data.
    """
    # Resample all the data frames.
    df_resample_list = []
    for df in df_list:
        # Convert the timestamp to datetime.
        df.index = pd.to_datetime(df.index, unit="s", utc=True)
        # Resample the timestamps by hour and average all the previous values.
        # Because we want data from the past, so label need to be "right".
        df_resample_list.append(df.resample("60Min", label="right").mean())

    # Merge all data frames.
    df = df_resample_list.pop(0)
    index_name = df.index.name
    while len(df_resample_list) != 0:
        # We need to use outer merging since we want to preserve data from both data frames.
        df = pd.merge_ordered(df, df_resample_list.pop(0), on=df.index.name, how="outer", fill_method=None)
        # Move the datetime column to index
        df = df.set_index(index_name)

    # Fill in the missing data with value -1.
    df = df.fillna(-1)
    return df


def answer_preprocess_smell(df):
    """
    This function is the answer of task 4.
    Preprocess smell data.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw smell reports data.

    Returns
    -------
    pandas.DataFrame
        The preprocessed smell data.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Drop the columns that we do not need.
    df = df.drop(columns=["feelings_symptoms", "smell_description", "zipcode"])

    # Select only the reports within the range of 3 and 5.
    df = df[(df["smell_value"]>=3)&(df["smell_value"]<=5)]

    # Convert the timestamp to datetime.
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    # Resample the timestamps by hour and sum up all the future values.
    # Because we want data from the future, so label need to be "left".
    df = df.resample("60Min", label="left").sum()

    # Fill in the missing data with value 0.
    df = df.fillna(0)
    return df


def answer_sum_current_and_future_data(df, n_hr=0):
    """
    This function is the answer of task 6.
    Sum up data in the current and future hours.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed smell data.
    n_hr : int
         Number of hours that we want to sum up the future smell data.

    Returns
    -------
    pandas.DataFrame
        The transformed smell data.
    """
    # Copy data frame to prevent editing the original one.
    df = df.copy(deep=True)

    # Fast return if n_hr is 0
    if n_hr == 0: return df

    # Sum up all smell_values in future hours.
    # The rolling function only works for summing up previous values.
    # So we need to shift back to get the value in the future.
    # Be careful that we need to add 1 to the rolling window size.
    # Becasue window size 1 means only using the current data.
    # Parameter "closed" need to be "right" because we want the current data.
    df = df.rolling(n_hr+1, min_periods=1, closed="right").sum().shift(-1*n_hr)

    # Below is an alternative implementation of rolling.
    #indexer = FixedForwardWindowIndexer(window_size=n_hr+1)
    #df = df.rolling(indexer, min_periods=1).sum()

    # Delete the last n_hr rows.
    # These n_hr rows have wrong data due to data shifting.
    df = df.iloc[:-1*n_hr]
    return df


def insert_previous_data_to_cols(df, n_hr=0):
    """
    Insert columns to indicate the data from the previous hours.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed sensor data.
    n_hr : int
        Number of hours that we want to insert the previous sensor data.

    Returns
    -------
    pandas.DataFrame
        The transformed sensor data.
    """
    # Copy data frame to prevent editing the original one.
    df = df.copy(deep=True)

    # Add the data from the previous hours.
    df_all = []
    for h in range(1, n_hr + 1):
        # Shift the data frame to get previous data.
        df_pre = df.shift(h)
        # Edit the name to indicate it is previous data.
        # The orginal data frame already has data from the previous 1 hour.
        # (as indicated in the preprocessing phase of sensor data)
        # So we need to add 1 here.
        df_pre.columns += "_pre_" + str(h+1) + "h"
        # Add the data to an array for merging.
        df_all.append(df_pre)

    # Rename the columns in the original data frame.
    # The orginal data frame already has data from the previous 1 hour.
    # (as indicated in the preprocessing phase of sensor data)
    df.columns += "_pre_1h"

    # Merge all data.
    df_merge = df
    for d in df_all:
        # The join function merges dataframes by index.
        df_merge = df_merge.join(d)

    # Delete the first n_hr rows.
    # These n_hr rows have no data due to data shifting.
    df_merge = df_merge.iloc[n_hr:]
    return df_merge


def convert_wind_direction(df):
    """
    Convert wind directions to sine and cosine components.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame that contains the wind direction data.

    Returns
    -------
    pandas.DataFrame
        The transformed data frame.
    """
    # Copy data frame to prevent editing the original one.
    df_cp = df.copy(deep=True)

    # Convert columns with wind directions.
    for c in df.columns:
        if "SONICWD_DEG" in c:
            df_c = df[c]
            df_c_cos = np.cos(np.deg2rad(df_c))
            df_c_sin = np.sin(np.deg2rad(df_c))
            df_c_cos.name += "_cosine"
            df_c_sin.name += "_sine"
            df_cp.drop([c], axis=1, inplace=True)
            df_cp[df_c_cos.name] = df_c_cos
            df_cp[df_c_sin.name] = df_c_sin
    return df_cp


def compute_feature_label(df_smell, df_sensor, b_hr_sensor=0, f_hr_smell=0):
    """
    Compute features and labels from the smell and sensor data.

    Parameters
    ----------
    df_smell : pandas.DataFrame
        The preprocessed smell data.
    df_sensor : pandas.DataFrame
        The preprocessed sensor data.
    b_hr_sensor : int
        Number of hours that we want to insert the previous sensor data.
    f_hr_smell : int
        Number of hours that we want to sum up the future smell data.

    Returns
    -------
    df_x : pandas.DataFrame
        The features that we want to use for modeling.
    df_y : pandas.DataFrame
        The labels that we want to use for modeling.
    """
    # Copy data frames to prevent editing the original ones.
    df_smell = df_smell.copy(deep=True)
    df_sensor = df_sensor.copy(deep=True)

    # Replace -1 values in sensor data to NaN
    df_sensor[df_sensor==-1] = np.nan

    # Convert all wind directions.
    df_sensor = convert_wind_direction(df_sensor)

    # Scale sensor data and fill in missing values
    df_sensor = (df_sensor - df_sensor.mean()) / df_sensor.std()
    df_sensor = df_sensor.round(6)
    df_sensor = df_sensor.fillna(-1)

    # Insert previous sensor data as features.
    # Noice that the df_sensor is already using the previous data.
    # So b_hr_sensor=0 means using data from the previous 1 hour.
    # And b_hr_sensor=n means using data from the previous n+1 hours.
    df_sensor = insert_previous_data_to_cols(df_sensor, b_hr_sensor)

    # Sum up current and future smell values as label.
    # Notice that the df_smell is already the data from the future 1 hour.
    # (as indicated in the preprocessing phase of smell data)
    # So f_hr_smell=0 means using data from the future 1 hour.
    # And f_hr_smell=n means using data from the future n+1 hours.
    df_smell = answer_sum_current_and_future_data(df_smell, f_hr_smell)

    # Add suffix to the column name of the smell data to prevent confusion.
    # See the description above for the reason of adding 1 to the f_hr_smell.
    df_smell.columns += "_future_" + str(f_hr_smell+1) + "h"

    # We need to first merge these two timestamps based on the available data.
    # In this way, we synchronize the time stamps in the sensor and smell data.
    # This also means that the sensor and smell data have the same number of data points.
    df = pd.merge_ordered(df_sensor.reset_index(), df_smell.reset_index(), on=df_smell.index.name, how="inner", fill_method=None)

    # Sanity check: there should be no missing data.
    assert df.isna().sum().sum() == 0, "Error! There is missing data."

    # Separate features (x) and labels (y).
    df_x = df[df_sensor.columns].copy()
    df_y = df[df_smell.columns].copy()

    # Add the hour of day and the day of week.
    dow_radian = df["EpochTime"].dt.dayofweek.copy(deep=True) * 2 * np.pi / 6.0
    tod_radian = df["EpochTime"].dt.hour.copy(deep=True) * 2 * np.pi / 23.0
    df_x.loc[:,"day_of_week_sine"] = np.sin(dow_radian)
    df_x.loc[:,"day_of_week_cosine"] = np.cos(dow_radian)
    df_x.loc[:,"hour_of_day_sine"] = np.sin(tod_radian)
    df_x.loc[:,"hour_of_day_cosine"] = np.cos(tod_radian)
    return df_x, df_y


# Now, we need to load and preprocess the dataset using the code in the structured data processing tutorial.

# In[5]:


# Load and preprocess sensor data
path = "smellpgh-v1/esdr_raw"
list_of_files = [f for f in listdir(path) if isfile(join(path, f))]
sensor_raw_list = []
for f in list_of_files:
    sensor_raw_list.append(pd.read_csv(join(path, f)).set_index("EpochTime"))
df_sensor = answer_preprocess_sensor(sensor_raw_list)

# Load and preprocess smell data
smell_raw = pd.read_csv("smellpgh-v1/smell_raw.csv").set_index("EpochTime")
df_smell = answer_preprocess_smell(smell_raw)

# Compute features and labels
df_x, df_y = compute_feature_label(df_smell, df_sensor, b_hr_sensor=2, f_hr_smell=7)

# Use value 40 as the threshold to indicate a smell event
# In this way, we make it a binary classification problem
df_y = (df_y>=40).astype(int)


# In[6]:


df_x


# In[7]:


df_y


# We now have the dataset ready in Pandas. To make it work for PyTorch, we need to first convert the `pandas.DataFrame` object to a `torch.Tensor` object with the `torch.float32` data type (because our PyTorch model will take this data type as the input). The PyTorch tensor object is similar to `numpy.array` but with more functions to support GPU (Graphics Processing Unit) computing. GPU can perform matrix operations much more efficiently than CPU (Central Processing Unit), and people usually use PyTorch to benefit from its powerful GPU computing support. The following code does the conversion. For more information about tensors, check this [UvA deep learning tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#Tensors) and this [PyTorch documentation](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html).

# In[8]:


# Convert pandas.DataFrame to torch.Tensor with type torch.float32
feature = torch.from_numpy(df_x.to_numpy()).float()
label = torch.from_numpy(df_y.to_numpy()).float()


# Next, we need to create a `torch.utils.data.Dataset` class, which is a uniform interface for loading data. More description about the PyTorch dataset class can be found in this [UvA deep learning tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#The-dataset-class) and this [PyTorch documentation](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class).

# In[9]:


class SmellPittsburghDataset(Dataset):
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


# We can use the dataset to create a training and validation set. In the code below, we use the first 8000 data points as the training set, followed by 168 data points as the validation set. We will use the training set to train the model and the validation set for evaluating model performance. In this case, 8000 data points mean 8000 hours, which is about 11 months in real-world time, and 168 data points represent about 7 days of time. The reason for doing this kind of split is because Smell Pittsburgh is a time-series dataset, which means we should use data from the past to predict the outcome in the future, but not the other way around.

# In[10]:


train_size = 8000
validation_size = 168
test_size = 168
i = 0
j = i + 8000
k = j + 168
dataset_train = SmellPittsburghDataset(feature=feature[i:j], label=label[i:j])
dataset_validation = SmellPittsburghDataset(feature=feature[j:k], label=label[j:k])


# Then, let us define a scoring function to evaluate the model performance for binary classification. The function takes two inputs: one is the array of predicted labels, and the other one is the true labels. In this case, we use precision, recall, and F1 score. Notice that we do a small trick here to return empty arrays of the output when the input is `None`, which will be handy later when we need to keep appending the scores into arrays of metrics.

# In[11]:


def binary_scorer(y_predict=None, y=None):
    """
    A customized scoring function to evaluate a binary classifier.

    Parameters
    ----------
    y_predict : torch.Tensor
        The predicted binary labels in 0 (negative) or 1 (positive).
    y : torch.Tensor
        The true binary labels in 0 (negative) or 1 (positive).

    Returns
    -------
    dict of int or float
        A dictionary of evaluation metrics.
    """
    if y_predict is not None and y is not None:
        # Compute metrics and return them
        eq = (y_predict == y)
        tp = (eq & (y_predict == 1)).sum()
        tn = (eq & (y_predict == 0)).sum()
        fp = (y_predict > y).sum()
        fn = (y_predict < y).sum()
        tpfp = tp + fp
        tpfn = tp + fn
        precision = 0 if tpfp == 0 else float(tp)/float(tpfp)
        recall = 0 if tpfn == 0 else float(tp)/float(tpfn)
        fscore = 0 if precision+recall==0 else 2*(precision*recall)/(precision+recall)
        return {"precision": precision, "recall": recall, "fscore": fscore}
    else:
        # Return the structure of the dictionary with empty arrays for initialization
        return {"precision": [], "recall": [], "fscore": []}


# Next, we need to train the model for multiple epochs, which means running through all the available data multiple times in the training set. Different from the traditional gradient descent, deep learning models use [Stochastic Gradient Descent (SGD) with mini-batches](http://d2l.ai/chapter_optimization/minibatch-sgd.html), which takes batches of data points (instead of all data). For more information about SGD, check [this StatQuest video](https://www.youtube.com/watch?v=vMh0zPT0tLI).
# 
# The reason for using mini-batch SGD is because training supervised deep learning models typically requires a lot of data (i.e., this is called many-shot learning now), and it is often impossible to fit all data into computer memory, so we have to feed the data to the optimization algorithm in batches. Also, training deep learning models usually requires going through all data points multiple times (i.e., multiple epochs). Batch size is a hyperparameter for tuning.
# 
# Now let us create a `torch.utils.data.DataLoader` object, which is a way to load data efficiently and is extremely beneficial if you cannot fit all data into computer memory at once (e.g., a bunch of videos). More information about DataLoader can be found in this [UvA deep learning tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#The-data-loader-class) or this [PyTorch documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders). The DataLoader object also allows us to load data in batches by specifying the batch size. Notice that we only want to shuffle the training set here, not the validation set. 
# 
# :::{important}
# When writing PyTorch code for training deep learning models, the first important thing is to get `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` objects ready.
# :::

# In[12]:


dataloader_train = DataLoader(dataset_train, batch_size=168, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=168, shuffle=False)


# To allow the DataLoader to use the device that we specified at the beginning of this notebook, we need to use the following code.

# In[13]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a chosen device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Now, we can move the DataLoader objects to the specified device.

# In[14]:


dataloader_train = DeviceDataLoader(dataloader_train, device)
dataloader_validation = DeviceDataLoader(dataloader_validation, device)


# Next, we need to define the deep regression model. We use two layers of linear neurons. The first layer maps the features to 64 hidden units (i.e., linear neurons), and the second layer maps 64 hidden units to one single output (i.e., whether there is a smell event or not).
# 
# Notice that for computational efficiency, we do not need to ensure that the output is probability since the loss function that we will define later already does this job for us (i.e., the [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)). We only need to make sure that it is probability later when we are going to use the model to make predictions of events for our task.
# 
# In the forward function, we define how the network will pass the result from each layer to the next layer. We use the ReLU activation function between the first and second layers to introduce some non-linearity. We do not need to define a backward function since PyTorch can automatically compute and backpropagate the gradients for us to iteratively update the model weights.
# 
# In this case, we choose to use 64 hidden units for demonstration purposes. In reality, this is a hyperparameter that you can tune. The input size should match the number of features (otherwise, running the code will give an error).

# In[15]:


class DeepRegression(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(DeepRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


# We are getting close to being able to train the model. The last three things that we need are to create the model object, specify the loss function (for the optimization algorithm to compute the error so that PyTorch can compute and backpropagate the gradients to update the model weights), and define the optimizer (i.e., the optimization algorithm).
# 
# Regarding the model, the code above defines a class. To be able to use it, we need to use the class to create an object. Think about a class as a design specification (e.g., spec for a car) and an object as producing the real design artifact (e.g., a real car). Regarding the loss criterion, we use the [Binary Cross Entropy (BCE) loss function](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#Loss-modules), which is standard for binary classification. Notice that we are not using [torch.nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) for computational efficiency. The [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) that we use instead here can take care of the job of transforming the output of the final neural network layer into probabilities.
# 
# Regarding the optimizer, we use the [Adam optimization algorithm](https://arxiv.org/pdf/1412.6980.pdf), which is a variation of Stochastic Gradient Descent with advanced capabilities in scheduling learning rates and scaling the gradients dynamically. Here, we use `0.0005` as the learning rate and `0.0001` as the weight decay for regularization. Adding the regularization can make the training more stable and prevent overfitting. In reality, they are hyperparameters for tuning.

# In[16]:


model = DeepRegression(feature.shape[1])
model = to_device(model, device) # move the model to the specified device
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)


# Finally, we can use the following function to train the model for multiple epochs and return the model performance metrics. We will use the [`tqdm` package](https://tqdm.github.io/) for help in tracking the progress in the for-loop, which is handy for deep learning code that can take a long time to run.
# 
# As mentioned before, the output of the model is not probability for computational efficiency. But when computing the model performance, we need the probability to determine how likely there will be smell events. We will use the Sigmoid function to convert the output from the final layer to probabilities that sum up to one. Then, we can check if the probability is larger than a threshold (e.g., 0.5).

# In[17]:


def train_model(dataloader, model, criterion, optimizer, num_epochs):
    """
    Train a PyTorch model and print model performance metrics.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader object.
    model : torch.nn.Module
        The PyTorch model that we want to train.
    criterion : torch.nn.modules.loss._Loss
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimization algorithm.
    num_epochs : int
        Number of epochs for training the model.
    """
    model.train() # set the model to training mode
    # Loop epochs
    for epoch in tqdm(range(num_epochs)):
        score_arrays = binary_scorer() # get the empty structure
        score_arrays["loss"] = [] # add the field for appending the loss
        # Loop the dataloader
        for x, y in dataloader:
            y_pred = model(x) # make predictions using the model
            loss = criterion(y_pred, y) # compute the loss
            optimizer.zero_grad() # set initial gradients to zero
            loss.backward() # accumulate and backpropagate the gradients
            optimizer.step() # update model weights
            score_arrays["loss"].append(loss.item()) # append the loss
            y_label = torch.sigmoid(y_pred) # turn model output into probabilities
            y_label = (y_pred > 0.5).float() # turn probabilities to labels with 0 or 1
            score = binary_scorer(y_label, y) # compute evaluation metrics
            # Append the evaluation metrics to the arrays
            for k in score:
                score_arrays[k].append(score[k])
        # After every 10 epochs, print the averaged evaluation metrics
        if epoch % 10 == 0:
            print("="*40)
            print("Epoch %d" % epoch)
            for k in score_arrays:
                print("averaged training %s: %4f" % (k, np.mean(score_arrays[k])))


# Next, we need a function to evaluate the model on the valiation or test set.

# In[18]:


def evaluate_model(dataloader, model):
    """
    Evaluate a PyTorch model and print model performance metrics.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader object.
    model : torch.nn.Module
        The PyTorch model that we want to train.
    """
    model.eval() # set the model to evaluation mode
    # Since we do not want to train the model, make sure that we deactivate the gradients
    with torch.no_grad():
        score_arrays = binary_scorer() # get the empty structure
        # Loop the dataloader
        for x, y in dataloader:
            y_pred = model(x) # make predictions using the model
            y_label = torch.sigmoid(y_pred) # turn model output into probabilities
            y_label = (y_pred > 0.5).float() # turn probabilities to labels with 0 or 1
            score = binary_scorer(y_label, y) # compute evaluation metrics
            # Append the evaluation metrics to the arrays
            for k in score:
                score_arrays[k].append(score[k])
    # Print the averaged evaluation metrics
    print("="*40)
    print("="*40)
    print("Validate model performance:")
    for k in score_arrays:
        print("averaged validation %s: %4f" % (k, np.mean(score_arrays[k])))


# Finally, we can now run the functions to train the model for 30 epochs and evaluate the model on the validation set. In practice, you can run the model for many epochs, save the model for every X epochs (e.g., X=5), and pick the model with the highest performance on the validation set.

# In[19]:


train_model(dataloader_train, model, criterion, optimizer, 30)
evaluate_model(dataloader_train, model)


# ## Your Task

# We have used the deep regression model to perform smell event classification on the Smell Pittsburgh dataset. Now, your task is to implement cross-validation (similar to the one that is used in the structured data processing tutorial) and perform hyperparameter tuning. Write your code below and print the averaged precision, recall, and F1 score after performing cross-validation and hyperparameter tuning. There are four hyperparameters that you can tune: the hidden unit size of the model, learning rate, weight decay, and batch size.
# 
# Notice that you also need to allocate a test set to have a more objective estimation of model performance. The test set cannot be seen by the model during hyperparameter tuning. Report also the precision, recall, and F1 score for the test set.
# 
# Optionally, you can also change the model architecture, such as adding more layers or changing the activation function. Have fun playing with deep learning models!

# In[20]:


# Write your code here

