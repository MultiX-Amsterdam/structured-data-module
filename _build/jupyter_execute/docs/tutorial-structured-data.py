#!/usr/bin/env python
# coding: utf-8

# # Tutorial (Structured Data Processing)

# (Last updated: Jan 24, 2025)[^credit]
# 
# [^credit]: Credit: this teaching material is created by [Yen-Chia Hsu](https://github.com/yenchiah) and revised by [Alejandro Monroy](https://github.com/amonroym99).

# This tutorial will familiarize you with the data science pipeline of processing structured data, using a real-world example of building models to predict and explain the presence of bad smell events in Pittsburgh based on air quality and weather data. The models are used to send push notifications about bad smell events to inform citizens, as well as to explain local pollution patterns to inform stakeholders.
# 
# 
# The scenario is in the next section of this tutorial, and more details are in the introduction section of the [Smell Pittsburgh paper](https://doi.org/10.1145/3369397). We will use the [same dataset as used in the Smell Pittsburgh paper](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction/tree/master/dataset/v1) as an example of structured data. During this tutorial, we will explain what the variables in the dataset mean and also guide you through model building. Below is the pipeline of this tutorial.

# \
# <img src="images/smellpgh-pipeline.png" style="max-width: 700px;">

# \
# You can use the following link to jump to the tasks and assignments:
# - [Task 4: Preprocess Sensor Data](#t4)
#   - [Assignment for Task 4](#a4)
# - [Task 5: Preprocess Smell Data](#t5)
#   - [Assignment for Task 5](#a5)
# - [Task 6: Prepare Features and Labels](#t6)
#   - [Assignment for Task 6](#a6)
# - [Task 7: Train and Evaluate Models](#t7)
#   - [Principal Component Analysis](#pca)
#   - [Evaluation Metrics](#evaluation-metrics)
#   - [Use the Dummy Classifier](#dummy-classifier)
#   - [Use the Decision Tree Model](#decision-tree)
#   - [Use the Random Forest Model](#random-forest)
#   - [Compute Feature Importance](#feature-importance)
#   - [Assignment for Task 7](#a7)

# ## Scenario

# Local citizens in Pittsburgh are organizing communities to advocate for changes in air pollution regulations. Their goal is to investigate the air pollution patterns in the city to understand the potential sources related to the bad odor. The communities rely on the Smell Pittsburgh application (as indicated in the figure below) to collect smell reports from citizens that live in the Pittsburgh region. Also, there are air quality and weather monitoring stations in the Pittsburgh city region that provide sensor measurements, including common air pollutants and wind information.

# \
# <img src="images/smellpgh-ui.png" style="max-width: 700px;">

# \
# You work in a data science team to develop models to map the sensor data to bad smell events. Your team has been working with the Pittsburgh local citizens closely for a long time, and therefore you know the meaning of each variable in the feature set that is used to train the machine learning model. The Pittsburgh community needs your help timely to analyze the data that can help them present evidence of air pollution to the municipality and explain the patterns to the general public.

# ## Import Packages

# :::{important}
# You need to install the packages [in this link](https://github.com/MultiX-Amsterdam/structured-data-module/blob/main/install_packages.sh) in your Python development environment.
# :::
# 
# We put all the packages that are needed for this tutorial below:

# Set the global style of all Pandas table output:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from os.path import isfile, join
from os import listdir
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Import the answers for the tasks
from util.answer import (
   answer_preprocess_sensor,
   answer_preprocess_smell,
   answer_sum_current_and_future_data,
   answer_experiment
)

# Import the utility functions that are provided
from util.util import (
    check_answer_df,
    compute_feature_importance,
    train_and_evaluate,
    plot_smell_by_day_and_hour,
    convert_wind_direction,
    insert_previous_data_to_cols,
    get_pca_result
)


# In[2]:


from IPython.core.display import HTML

def set_global_df_style():
    styles = """
    <style>
        .dataframe * {font-size: 1em !important;}
    </style>
    """
    return HTML(styles)

# Call this function in a notebook cell to apply the style globally
set_global_df_style()


# <a name="answer"></a>

# ## Task Answers

# Click on one of the following links to check answers for the assignments in this tutorial. **Do not check the answers before practicing the tasks.**
# - [Click this for task answers if you open this notebook on your local machine](util/answer.py)
# - {any}`Click this for task answers if you view this notebook on a web browser <./answer>`

# <a name="util"></a>

# ## Utility File

# Click on one of the following links to check the provided functions in the utility file for this tutorial.
# - [Click this for utility functions if you open this notebook on your local machine](util/util.py)
# - {any}`Click this for utility functions if you view this notebook on a web browser <./util>`

# <a name="t4"></a>

# ## Task 4: Preprocess Sensor Data

# In this task, we will process the sensor data from various air quality monitoring stations in Pittsburgh. First, we need to load all the sensor data.

# In[3]:


path = "smellpgh-v1/esdr_raw"
list_of_files = [f for f in listdir(path) if isfile(join(path, f))]
sensor_raw_list = []
for f in list_of_files:
    sensor_raw_list.append(pd.read_csv(join(path, f)).set_index("EpochTime"))


# Now, the `sensor_raw_list` variable contains all the data frames with sensor values from different air quality monitoring stations. Noted that `sensor_raw_list` is an array of data frames. We can print one of them to take a look, as shown below. 

# In[4]:


sensor_raw_list[0]


# The `EpochTime` index is the timestamp in epoch time, which means the number of seconds that have elapsed since January 1st, 1970 (midnight UTC/GMT). Other columns mean the sensor data from an air quality monitoring station.

# Next, we need to resample and merge all the sensor data frames so that they can be used for modeling. Our goal is to have a dataframe that looks like the following:

# In[5]:


df_sensor = answer_preprocess_sensor(sensor_raw_list)
df_sensor


# In the expected output above, the `EpochTime` index is converted from timestamps into [pandas datetime](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex) objects, which has the format `year-month-day hour:minute:second+timezone`. The `+00:00` string means the GMT/UTC timezone. Other columns mean the average value of the sensor data in the previous hour. For example, `2016-10-31 06:00:00+00:00` means October 31 in 2016 at 6AM UTC time, and the cell with column `3.feed_1.SO2_PPM` means the averaged SO2 (sulfur dioxide) values from 5:00 to 6:00.
# 
# The column name suffix `SO2_PPM` means sulfur dioxide in unit PPM (parts per million). The prefix `3.feed_1.` in the column name means a specific sensor (feed ID 1). You can ignore the `3.` at the begining of the column name. You can find the list of sensors, their names with feed ID (which will be in the data frame columns), and also the meaning of all the suffixes from [this link](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction/tree/master/dataset/v2.1#description-of-the-air-quality-sensor-data).
# 
# Some column names look like `3.feed_11067.SIGTHETA_DEG..3.feed_43.SIGTHETA_DEG`. This means that the column has data from two sensor stations (feed ID 11067 and 43). The reason is that some sensor stations are replaced by the new ones over time. So in this case, we merge sensor readings from both feed ID 11067 and 43. 

# <a name="a4"></a>

# ### Assignment for Task 4

# **Your task (which is your assignment) is to write a function to do the following:**
# - Sensors can report in various frequencies. So, for each data frame, we need to resample the data by computing the hourly average of sensor measurements from the "previous" hour. For example, at time 8:00, we want to know the average of sensor values between 7:00 and 8:00.
#   - Hint: Use the `pandas.to_datetime` function when converting timestamps to datetime objects. Type `?pd.to_datetime` in a code cell for more information.
#   - Hint: Use the `pandas.DataFrame.resample` function to resample data. Type `?pd.DataFrame.resample` in a code cell for more information.
#   - Hint: Use the `pandas.merge_ordered` function when merging data frames. Type `?pd.merge_ordered` in a code cell for more information.
# - Then, merge all the data frames based on their time stamp, which is the `EpochTime` column.
# - Finally, fill in the missing data with the value -1. The reason for not using 0 is that we want the model to know if sensors give values (including zero) or no data.
#   - Hint: Use the `pandas.DataFrame.fillna` function when treating missing values. Type `?pd.DataFrame.fillna` in a code cell for more information.
#  - Notice that you need to convert the column types in the dataframe to float to avoid errors when checking if the answers match the desired output. You can use the `pandas.DataFrame.astype` function to do this `df = df.astype(float)`.

# In[6]:


def preprocess_sensor(df_list):
    """
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
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The code below tests if the output of your function matches the expected output.

# In[7]:


check_answer_df(preprocess_sensor(sensor_raw_list), df_sensor, n=1)


# <a name="t5"></a>

# ## Task 5: Preprocess Smell Data

# In this task, we will preprocess the smell data. First, we need to load the raw smell data.

# In[8]:


smell_raw = pd.read_csv("smellpgh-v1/smell_raw.csv").set_index("EpochTime")
smell_raw


# The meaning of `EpochTime` is explained in the previous task. Other columns mean the self-reported symptoms, descriptions of smell, severity ratings (the `smell_value` column), and the zipcode where the report is submitted in Pittsburgh, Pennsylvania. For example, the second row means that the smell report was submitted from the 15218 zipcode with wood smoke description and severity rating 2. For more description about the smell, please check the [Smell Pittsburgh website](https://smellpgh.org/how_it_works).

# Next, we need to resample the smell data so that they can be used for modeling. Our goal is to have a dataframe that looks like the following:

# In[9]:


df_smell = answer_preprocess_smell(smell_raw)
df_smell


# In the latest row, the timestamp is `2018-09-30 04:00:00+00:00`, which means this row contains the data from 3:00 to 4:00 on September 30 in 2018. This row has `smell_value` 8, which means the sum of smell report ratings in the above mentioned time range. Notice that the expected output ignores all smell ratings from 1 to 2. This is becasue we only want the ratings that indicate bad smell, which will be further explained below.

# <a name="a5"></a>

# ### Assignment for Task 5

# **Your task (which is your assignment) is to write a function to do the following:**
# - First, remove the `feelings_symptoms`, `smell_description`, and `zipcode` columns since we do not need them.
#   - Hint: Use the `pandas.DataFrame.drop` function. Type `?pd.DataFrame.drop` in a code cell for more information.
# - We only want the reports that indicate bad smell. You need to select only the reports with rating 3, 4, or 5 in the `smell_value` column.
# - Then, we want to know the severity of bad smell within an hour in the future. For example, at time 8:00, we want to know the sum of smell values between 8:00 and 9:00. So you need to resample the data by computing the hourly sum of smell values from the "future" hour.
#   - Hint: Use the `pandas.to_datetime` function when converting timestamps to datetime objects. Type `?pd.to_datetime` in a code cell for more information.
#   - Hint: Use the `pandas.DataFrame.resample` function to resample data. Type `?pd.DataFrame.resample` in a code cell for more information.
# - Finally, fill in the missing data with the value 0. The reason is that missing data means there are no smell reports (provided by citizens) within an hour, so we assume that there is no bad smell within this period of time. Notice that this is an assumption and also a limitation since citizens rarely report good smell.
#   - Hint: Use the `pandas.DataFrame.fillna` function when treating missing values. Type `?pd.DataFrame.fillna` in a code cell for more information.
#  - Notice that you need to convert the column types in the dataframe to float to avoid errors when checking if the answers match the desired output. You can use the `pandas.DataFrame.astype` function to do this `df = df.astype(float)`.

# In[10]:


def preprocess_smell(df):
    """
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
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The code below tests if the output of your function matches the expected output.

# In[11]:


check_answer_df(preprocess_smell(smell_raw), df_smell, n=1)


# Now, we can plot the distribution of smell values by using the `pandas.DataFrame.plot` function.

# In[12]:


fig = df_smell.plot(kind="hist", bins=20, ylim=(0,100), edgecolor="black").set_yticks([0,50,100], labels=["0","50",">100"])


# From the plot above, we can observe that a lot of the time, the smell values are fairly low. This means that smell events only happen occasionally, and thus our dataset is highly imbalanced.

# We can also plot the average number of smell reports distributed by the day of week (Sunday to Saturday) and the hour of day (0 to 23), using a provided function `plot_smell_by_day_and_hour` in the [utility file](#util).

# In[13]:


plot_smell_by_day_and_hour(df_smell)


# From the plot above, we can observe that citizens tend to report smell in the morning.

# <a name="t6"></a>

# ## Task 6: Prepare Features and Labels

# Now we have the preprocessed data in the `df_sensor` and `df_smell` variables. Our next task is to prepare features and labels for modeling, as shown in the figure below.

# \
# <img src="images/smellpgh-predict.png" style="max-width: 700px;">

# \
# Our goal is to construct two data frames, `df_x` and `df_y`, that represent the features and labels, respectively. First, we will deal with the sensor data. We need a function to insert columns (that indicate previous `n_hr` hours of sensor) to the existing data frame, where `n_hr` should be a parameter that we can control. We provide the `insert_previous_data_to_cols` function in the [utility file](#util) to do this.

# The code below shows a test case, which is a part of the sensor data.

# In[14]:


# Below is an example input.
df_sensor_example_in = df_sensor[["3.feed_1.SONICWS_MPH"]][0:5]
df_sensor_example_in


# In[15]:


# Below is the expected output of the above example input.
df_sensor_example_out = insert_previous_data_to_cols(df_sensor_example_in, n_hr=2)
df_sensor_example_out


# The reason that there are 2 less rows in the expected output is because we set `n_hr=2`, which means there are missing data in the original first and second row (because there was no previous data for these rows). So in the code, we removed these rows.

# Notice that the `insert_previous_data_to_cols` function added suffixes to the column names to indicate the number of hours that the sensor measurements came from previously. Pay attention to the meaning of time range here.
# - For example, in the first row, the `3.feed_1.SONICWS_MPH_pre_1h` column has value `3.4`, which means the average reading of wind speed (in unit MPH) between the current time stamp (which is `8:00`) and the previous 1 hour (which is `7:00`).
# - In the second column of the first row, the `3.feed_1.SONICWS_MPH_pre_2h` has value `3.5`, which means the average reading of wind speed between the previous 1 hour (which is `7:00`) and 2 hours (which is `6:00`).
# - It is important to note here that suffix `pre_2h` does **not** mean the average rating within 2 hours between the current time stamp and the time that is 2 hours ago.

# Then, we also need a function to convert wind direction into sine and cosine components, which is a common technique for encoding cyclical features (i.e., any that that circulates within a set of values, such as hours of the day, days of the week). The formula is below:
# 
# $$
# x_{sin} = \sin{\Big(\frac{2 \cdot \pi \cdot x}{\max{(x)}}\Big)}
# \\
# x_{cos} = \cos{\Big(\frac{2 \cdot \pi \cdot x}{\max{(x)}}\Big)}
# $$
# 
# There are several reasons to do this instead of using the original wind direction degrees (that range from 0 to 360). First, by applying sine and cosine to the degrees, we can transform the original data to a continuous variable. The original data is not continuous since there are no values below 0 or above 360, and there is no information to tell that 0 degrees and 360 degrees are the same. Second, the decomposed sine and cosine components allow us to inspect the effect of wind on the north-south and east-west directions separately, which may help us explain the importance of wind directions. See the code for function `convert_wind_direction` in the [utility file](#util) for achieving this.

# The code below shows a test case, which is a part of the sensor data.

# In[16]:


# Below is an example input.
df_wind_example_in = df_sensor[["3.feed_1.SONICWD_DEG"]][0:5]
df_wind_example_in


# In[17]:


# Below is the expected output of the above example input.
df_wind_example_out = convert_wind_direction(df_wind_example_in)
df_wind_example_out


# We have dealt with the sensor data. Next, we will deal with the smell data. We need a function to sum up smell values in the future hours, where `n_hr` should be a parameter that we can control. The code below shows a test case, which is a part of the smell data.

# In[18]:


# Below is an example input.
df_smell_example_in = df_smell[107:112]
df_smell_example_in


# In[19]:


# Below is the expected output of the above example input.
df_smell_example_out1 = answer_sum_current_and_future_data(df_smell_example_in, n_hr=1)
df_smell_example_out1


# In[20]:


# Below is another expected output with a different n_hr.
df_smell_example_out2 = answer_sum_current_and_future_data(df_smell_example_in, n_hr=3)
df_smell_example_out2


# In the output above, notice that row `2016-11-05 10:00:00+00:00` has smell value `83`, and the setting is `n_hr=3`, which means the sum of smell values within `n_hr+1` hours (i.e., from `10:00` to `14:00`) is 83. Pay attention to this setup since it can be confusing. The reason of `n_hr+1` (but not `n_hr`) is because the input data already indicates the sum of smell values within the future 1 hour.

# In[21]:


# Below is another expected output when n_hr is 0.
df_smell_example_out3 = answer_sum_current_and_future_data(df_smell_example_in, n_hr=0)
df_smell_example_out3


# <a name="a6"></a>

# ### Assignment for Task 6

# **Your task (which is your assignment) is to write a function to do the following:**
# - First, perform a [windowing operation](https://pandas.pydata.org/docs/user_guide/window.html#rolling-window-endpoints) to sum up smell values within a specified `n_hr` time window.
#   - Hint: Use the `pandas.DataFrame.rolling` function when summing up values within a window. Type `?pd.DataFrame.rolling` in a code cell for more information.
#   - Hint: Use the `pandas.DataFrame.shift` fuction to shift the rolled data back `n_hr` hours because we want to sum up the values in the future (the rolling function operates on the values in the past). Type `?pd.DataFrame.shift` in a code cell for more information.
# - Finally, Remove the last `n_hr` hours of data because they have the wrong data due to shifting. For example, the last row does not have data in the future to operate if we set `n_hr=1`.
#   - Hint: Use `pandas.DataFrame.iloc` to select the rows that you want to keep. Type `?pd.DataFrame.iloc` in a code cell for more information.
# - You need to handle the edge case when `n_hr=0`, which should output the original data frame.
#  - Notice that you need to convert the column types in the dataframe to float to avoid errors when checking if the answers match the desired output. You can use the `pandas.DataFrame.astype` function to do this `df = df.astype(float)`.

# In[22]:


def sum_current_and_future_data(df, n_hr=0):
    """
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
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The code below tests if the output of your function matches the expected output.

# In[23]:


check_answer_df(sum_current_and_future_data(df_smell_example_in, n_hr=1), df_smell_example_out1, n=1)


# In[24]:


check_answer_df(sum_current_and_future_data(df_smell_example_in, n_hr=3), df_smell_example_out2, n=2)


# In[25]:


check_answer_df(sum_current_and_future_data(df_smell_example_in, n_hr=0), df_smell_example_out3, n=3)


# Finally, we need a function to compute the features and labels, based on the `insert_previous_data_to_cols` and `sum_current_and_future_data` functions. The code is below.

# In[26]:


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

    return df_x, df_y


# We will use the sensor data within the previous 3 hours to predict bad smell within the future 8 hours. To use the `compute_feature_label` function that we just build, we need to set `b_hr_sensor=2` and `f_hr_smell=7` because originally `df_sensor` already contains data from the previous 1 hour, and `df_smell` already contains data from the future 1 hour.
# 
# Note that `b_hr_sensor=n` means that we want to insert previous `n+1` hours of sensor data , and `f_hr_smell=m` means that we want to sum up the smell values of the future `m+1` hours. For example, suppose that the current time is 8:00, setting `b_hr_sensor=2` means that we use all sensor data from 5:00 to 8:00 (as features `df_x` in prediction), and setting `f_hr_smell=7` means that we sum up the smell values from 8:00 to 16:00 (as labels `df_y` in prediction).

# In[27]:


df_x, df_y = compute_feature_label(df_smell, df_sensor, b_hr_sensor=2, f_hr_smell=7)


# Below is the data frame of features (i.e., the predictor variable). There are negative float numbers because the `compute_feature_label` function scales the sensor data by subtracting its mean value and dividing it by the standard deviation.

# In[28]:


df_x


# Below is the data frame of labels (i.e., the response variable).

# In[29]:


df_y


# <a name="t7"></a>

# ## Task 7: Train and Evaluate Models

# We have processed raw data and prepared the `compute_feature_label` function to convert smell and sensor data into features `df_x` and labels `df_y`. In this task, you will work on training basic and advanced models to predict bad smell events (i.e., the situation that the smell value is high) using the sensor data from air quality monitoring stations.

# First, let us threshold the features to make them binary for our classification task. We will use value 40 as the threshold to indicate a smell event. The threshold 40 was used in the Smell Pittsburgh research paper. It is equivalent to the situation that 10 people reported smell with rating 4 within 8 hours.

# In[30]:


df_y_40 = (df_y>=40).astype(int)
df_y_40


# The descriptive statistics below tell us that the dataset is imbalanced, which means that the numbers of data points in the positive and negative label groups have a big difference.

# In[31]:


print("There are %d rows with smell events." % (df_y_40.sum().iloc[0]))
print("This means %.2f proportion of the data has smell events." % (df_y_40.sum().iloc[0]/len(df_y_40)))


# <a name="pca"></a>

# ### Principal Component Analysis

# We can also plot the result from the Principal Component Analysis (PCA) to check if there are patterns in the data. 
# 
# PCA is a linear dimensionality reduction technique aiming to transform the original variables into a new set of uncorrelated variables (principal components) that preserve as much information or variability in the dataset as possible. To do so, it applies a linear transformation to the data onto a new coordinate system such that the directions (principal components) that capture the largest variation in the data can be identified. 
# 
# If you are interested in the mathematical explanation of PCA, you can check the following hidden block. Otherwise, please feel free to skip the math.

# :::{admonition} Click the button to show the math for PCA
# :class: dropdown note
# 
# Principal components are the eigenvectors of the covariance matrix of the centered data. The magnitude of their corresponding eigenvalues indicates the explained variance that each principal component captures. That way, the eigenvectors with the $n$ highest principal components will be chosen as principal components.
# 
# The step to apply PCA with $n$ components to a dataset $X$ are the following:
# 
# 1. **Calculate covariance matrix of the centered data:** 
# 
#     Subtract the mean ($\mu_X$) of each feature
# 
#     $$ \bar{X} = X - \mu_X, $$
#     
#     and compute the covariance matrix of the centered data $\bar{X}$:
#     
#     $$ C = \frac{1}{N} \bar{X}^T \bar{X}.$$
#     
# 2. **Compute Eigenvalues and Eigenvectors:** 
# 
#     Derive the eigenvalues $\{\lambda_i\}$ and eigenvectors $\{v_i\}$ of the covariance matrix $C$. Each eigenvector represents a principal component, and its corresponding eigenvalue indicates the variance explained by that principal component.
# 
# 3. **Choose the principal components** 
# 
#     Order the eigenvalues in descending order.
#     
#     $$ \lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_D $$
# 
#     and select $v_1$, $v_2$, ..., $v_n$ as principal components.
# 
# 4. **Data Transformation:** 
# 
#     Use the original data $X$ and the chosen principal components to obtain the transformed data $X'$:
#     
#     $$ X' = X P_k $$
#     
#     where $P_n$ is the matrix with the first $n$ eigenvectors as columns.
# :::

# Dimensionality reduction techiniques are often used to visualize larger-dimensional data in a 2D or 3D space. Let's apply PCA to our example. 
# 
# The `get_pca_result` function in the [utility file](#util) creates a dataframe with the `n` principal components in the dataset.

# The code below prints the outputs from the `get_pca_result` function.

# In[32]:


df_pc_pca, df_r_pca = get_pca_result(df_x, df_y_40, n=3)
print(df_pc_pca)
print(df_r_pca)


# Now let us plot the ratios of explained variances (i.e., the eigenvalues). The intuition is that if the explained variance ratio is larger, the corresponding principal component is more important and can represent more information.

# In[33]:


# Plot the eigenvalues.
ax1 = sns.barplot(x="pc",y="var", data=df_r_pca, color="c")
ax1 = ax1.set(xlabel="Principal Component",
       ylabel="Ratio of Variance Explained",
       title="Variance Explained by Principal Components")


# We can also plot the first and second principal components to see the distribution of labels. In the figure below, we see that label `1` (which means having a smell event) is concentrated on one side. However, it looks like labels `0` and `1` do not separate well, which means that the classifier will have difficulty distinguishing these two groups.

# In[34]:


# Plot two principal components.
ax2 = sns.lmplot(x="PC1", y="PC2", data=df_pc_pca,
    fit_reg=False, hue="y", legend=True,
    scatter_kws={"s": 10, "alpha" :0.4})
ax2 = ax2.set(title="Distribution of Labels by Principal Components")


# In the interactive figure below, we can zoom in, zoom out, and rotate the view to check the distribution of labels. We can confirm that label `1` is concentrated on one side. But we also observe that there is a large part of the data that may not be separated well by the classifier (especially linear classifiers). We probably need non-linear classifiers to be able to tell the differences between these two groups.

# In[35]:


# Use the Plotly package to show the PCA results.
fig = px.scatter_3d(df_pc_pca.sample(n=5000), x="PC1", y="PC2", z="PC3",
                    color="y", symbol="y", opacity=0.6,
                    size="size", size_max=15,
                    category_orders=dict(y=["0", "1"]),
                    height=700)
fig.show()


# <a name="evaluation-metrics"></a>

# ### Evaluation Metrics

# Next, let us pick a subset of the sensor data (instead of using all of them) and prepare the function for computing the evaluation metrics. Our intuition is that the smell may come from chemical compounds near major pollution sources. From the knowledge of local people, there is a large pollution source, which is the Clairton Mill Works that belongs to the United States Steel Corporation. This pollution source is located at the south part of Pittsburgh. This factory produces petroleum coke, which is a fuel to refine steel. And during the coke refining process, it generates pollutants.
# 
# We think that H2S (hydrogen sulfide, smells like rotten eggs) and SO2 (sulfur dioxide, smells like burnt matches) near the pollution source may be good features. So we first select the columns with H2S and SO2 measurements from a monitoring station near this pollution source.

# In[36]:


df_x_subset = df_x[["3.feed_28.H2S_PPM_pre_1h", "3.feed_28.SO2_PPM_pre_1h"]]
df_x_subset


# Next, we will train and evaluate a model (F) that maps features (i.e., the sensor readings) to labels (i.e., the smell events). We have the `scorer` and `train_and_evaluate` functions ready to help us train and evaluate models.

# The `train_and_evaluate` function prints the averaged f1-score, averaged precision, averaged recall, and averaged accuracy across all the folds. These metrics are always in the range of zero and one, with zero being the worst and one being the best. We also printed the confusion matrix that contains true positives, false positives, true negatives, and false negatives. To understand the evaluation metrics, let us first take a look at the confusion matrix, explained below:
# - True Positives
#   - There is a smell event in the real world, and the model correctly predicts that there is a smell event.
# - False Positives
#   - There is no smell event in the real world, but the model falsely predicts that there is a smell event.
# - True Negatives
#   - There is no smell event in the real world, and the model correctly predicts that there is no smell event.
# - False Negatives
#   - There is a smell event in the real world, but the model falsely predicts that there is no smell event.
# 
# The accuracy metric is defined in the equation below:
# ```
# accuracy = (true positives + true negatives) / total number of data points
# ```
# 
# Accuracy is the number of correct predictions divided by the total number of data points. It is a good metric if the data distribution is not skewed (i.e., the number of data records that have a bad smell and do not have a bad smell is roughly equal). But if the data is skewed, which is the case in our dataset, we will need another set of evaluation metrics: f1-score, precision, and recall. We will use an example later to explain why accuracy is an unfair metric for our dataset.
# 
# The precision metric is defined in the equation below:
# ```
# precision = true positives / (true positives + false positives)
# ```
# 
# In other words, precision means how precise the prediction is. High precision means that if the model predicts “yes” for smell events, it is highly likely that the prediction is correct. We want high precision because we want the model to be as precise as possible when it says there will be smell events.
# 
# Next, the recall metric is defined in the equation below:
# ```
# recall = true positives / (true positives + false negatives)
# ```
# 
# In other words, recall means the ability of the model to catch events. High recall means that the model has a low chance to miss the events that happen in the real world. We want high recall because we want the model to catch all smell events without missing them.
# 
# Typically, there is a tradeoff between precision and recall, and one may need to choose to go for a high precision but low recall model, or we go for a high recall but low precision model. The tradeoff depends on the context. For example, in medical applications, one may not want to miss the events (e.g., cancer) since the events are connected to patients' quality of life. In our application of predicting smell events, we may not want the model to make false predictions when saying "yes" to smell events. The reason is that people may lose trust in the prediction model when we make real-world interventions incorrectly, such as sending push notifications to inform the users about the bad smell events.
# 
# The f1-score metric is a combination of recall and precision, as indicated below:
# ```
# f1-score = 2 * (precision * recall) / (precision + recall)
# ```

# <a name="dummy-classifier"></a>

# ### Use the Dummy Classifier

# Now that we have explained the evaluation metrics. Let us first use a dummy classifier that always predicts no smell events. In other words, the dummy classifier never predicts "yes" about the presence of smell events. Later we will guide you through using more advanced machine learning models. 

# In[37]:


dummy_model = DummyClassifier(strategy="constant", constant=0)
train_and_evaluate(dummy_model, df_x_subset, df_y_40, train_size=336, test_size=168)


# The printed message above shows the evaluation result of the dummy classifier. We see that the accuracy is 0.92, which is very high. But the f1-score, precision, and recall are zero since there are no true positives. This is because the Smell Pittsburgh dataset has a skewed distribution of smell events, which means that there are a lot of "no" (i.e., label `0`) but only a small part of "yes" (i.e., label `1`). This skewed data distribution corresponds to what happened in Pittsburgh. Most of the time, the odors in the city area are OK and not too bad. Occasionally, there can be very bad pollution odors, where many people complain.
# 
# By the definition of accuracy, the dummy classifier (which always says "no") has a very high accuracy of 0.92. This is because only 9% of the data indicate bad smell events. So, you can see that accuracy is not a fair evaluation metric for the Smell Pittsburgh dataset. And instead, we need to go for the f1-score, precision, and recall metrics.
# 
# This step uses cross-validation to evaluate the machine learning model, where the data is divided into several parts, and some parts are used for training. Other parts are used for testing. Typically people use K-fold cross-validation, which means that the entire dataset is split into K parts. One part is used for testing (i.e., the testing set), and the other parts are used for training (i.e., the training set). This procedure is repeated K times so that every fold has the chance of being tested. The result is averaged to indicate the performance of the model, for example, averaged accuracy. We can then compare the results for different machine learning pipelines.
# 
# However, the script uses a different cross-validation approach, where we only use the previous folds to train the model to test future folds. For example, if we want to test the third fold, we will only use a part of the data from the first and second fold to train the model. The reason is that the Smell Pittsburgh dataset is primarily time-series data, which means the dataset has timestamps for every data record. In other words, things that happened in the past may affect the future. So, in fact, it does not make sense to use the data in the future to train a model to predict what happened in the past. Our time-series cross-validation approach is shown in the following figure.

# \
# <img src="images/smellpgh-cross-validation.png" style="max-width: 700px;">

# \
# For the `train_and_evaluate` function, `test_size` is the number of samples for testing, and `train_size` is the number of samples for training. We need to set these numbers for time-series cross-validation. For example, setting `test_size` to 168 means using 168 samples for testing, which also means 168 hours (or 7 days) of data. Setting `train_size` to 336 means using 336 samples for testing, which also means 336 hours (or 14 days) of data. So, this means we are using previous 14 days of sensor data to train the model, and then use the model to predict the smell events in the next 7 days. In this setting, every Sunday we can re-train the model with the updated data, so that we have the updated model to predict smell events every week.

# <a name="decision-tree"></a>

# ### Use the Decision Tree Model

# Now, instead of using the dummy classifier, we are going to use a different model. Let us use the Decision Tree model and compare its performance with the dummy classifier.

# In[38]:


dt_model = DecisionTreeClassifier()
train_and_evaluate(dt_model, df_x_subset, df_y_40, train_size=336, test_size=168)


# From the printed message above, notice that the Decision Tree model produces non-zero true positives and false positives (compared to the dummy classifier). Also, notice that f1-score, precision, and recall are no longer zero.
# 
# Decision Tree is a type of machine learning model. You can think of it as how a medical doctor diagnoses patients. For example, to determine if the patients need treatments, the medical doctor may ask the patients to describe symptoms. Depending on the symptoms, the doctor decides which treatment should be applied for the patient.
# 
# One can think of the smell prediction model as an air quality expert who is diagnosing the pollution patterns based on air quality and weather data. The treatment is to send a push notification to citizens to inform them of the presence of bad smell to help people plan daily activities. This decision-making process can be seen as a tree structure as shown in the following figure, where the first node is the most important factor to decide the treatment.

# \
# <img src="images/smellpgh-decision-tree.png" style="max-width: 400px;">

# \
# The above figure is just a toy example to show what a decision tree is. In our case, we put the features (X) into the decision tree to train it. Then, the tree will decide which feature to use and what is the threshold to split the data based on the features. This procedure is repeated several times (represented by the depth of the tree). Finally, the model will make a prediction (y) at the final node of the tree.
# 
# You can find the visualization of a decision tree (trained using the real data) in Figure 8 in the [Smell Pittsburgh paper](https://doi.org/10.1145/3369397). More information about the Decision Tree can be found in the following link and paper:
# - [More information about Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
# - [Quinlan, J. R. (1986). Induction of decision trees. Machine learning, 1(1), 81-106.](https://doi.org/10.1007/BF00116251)

# <a name="random-forest"></a>

# ### Use the Random Forest Model

# Now that you have tried the Decision Tree model. Let us use a more advanced model, Random Forest, for smell event prediction.

# In[39]:


rf_model = RandomForestClassifier()
train_and_evaluate(rf_model, df_x_subset, df_y_40, train_size=336, test_size=168)


# Notice that the performance of the model does not look much better than the Decision Tree model. And in fact, both models currently have poor performance. This can have several meanings, as indicated in the following list. You will explore some of these questions in the assignment for this task.
# - Firstly, do we really believe that we are using a good set of features? Is it sufficient to only use the H2S and SO2 features? Is it sufficient to only include the data from the previous hour (i.e., the `"3.feed_28.H2S_PPM_pre_1h"` column)?
# - Secondly, the machine learning pipeline uses 14 days of data in the past (i.e., `train_size=336`) to predict smell events in the future 7 days (i.e., `test_size=168`). Do we believe that 14 days are sufficient for training a good model?
# - Finally, the [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) models has many hyper-parameters (e.g., maximum depth of tree). Currently, the model uses the default hyper-parameters. Do we believe that the default setting is good?
# 
# Now let us take a look at the Random Forest model, which is a type of ensemble model. You can think about the ensemble model as a committee that makes decisions collaboratively, such as using majority voting. For example, to determine the treatment of a patient, we can ask the committee of medical doctors for a collaborative decision. The committee has several members who correspond to various machine learning models. The Random Forest model is a committee that is formed with many decision trees. Each tree is trained using different sets of data, as shown in the following figure.

# \
# <img src="images/smellpgh-random-forest.png" style="max-width: 700px;">

# \
# In other words, we first trained many decision trees, and each of them has access to only a part of the data (but not all of the data) that are randomly selected. So, each decision tree sees different sets of features. Then, we ask the committee members (i.e., the decision trees) to make predictions, and the final result is the one that receives the highest votes.
# 
# The intuition for having a committee (instead of only a single tree) is that we believe a diverse set of models can make better decisions collaboratively. There is mathematical proof about this intuition, but the proof is outside the scope of this course. More information about the Random Forest model can be found in the following link and paper:
# - [More information about Random Forest](https://scikit-learn.org/stable/modules/ensemble.html)
# - [Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.](https://doi.org/10.1023/A:1010933404324)

# <a name="feature-importance"></a>

# ### Compute Feature Importance

# After training the model and evaluate its performance, we now have a better understanding about how to predict smell events. However, what if we want to know which are the important features? For example, which pollutants are the major source of the bad smell? Which pollution source is likely related to the bad smell? Under what situation will the pollutants travel to the Pittsburgh city area? This information can be important to help the municipality evaluate air pollution policies. This information can also help local communities to advocate for policy changes.
# 
# It turns out that we can permute the data in a specific column to know the importance. If a column (corresponding to a feature) is important, permuting the data specifically for the column will make the model performacne decrease. A higher decrease of a metric (e.g., f1-score) means that the feature is more important. It also means the feature is important for the model to make decisions. So, we can compute the "decrease" of a metric and use it as feature importance. We have provided a function `compute_feature_importance` in the [utility file](#util) to do this.

# In[40]:


compute_feature_importance(rf_model, df_x_subset, df_y_40, scoring="f1")


# **Notice that the `compute_feature_importance` function can take a very long time to run if you use a large amount of training data**.
# 
# The above code prints feature importance, which indicates the influence of each feature on the model prediction result. Here we use the Random Forest model. For example, in the printed message, the `3.feed_28.H2S_PPM_pre_1h` feature has the highest importance. The values in the feature importance here represent the decrease of f1-score (because we use `scoring="f1"`) if we randomly permute the data related to the feature.
# 
# For example, if we randomly permute the H2S measurement in the `3.feed_28.H2S_PPM_pre_1h` column for all the data records (but keep other features the same), the f1-score of the Random Forest model will drop about `0.39`. The intuition is that if a feature is more important, the model performance will decrease more when the feature is randomly permuted (i.e., when the data that is associated with the feature are messed up). More information can be found in the following link:
# - [More information about feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
# 
# Notice that to use this technique, the model needs to fit the data reasonably well. Also depending on the number of features you are using, the step of computing the feature importance can take a lot of time.

# <a name="a7"></a>

# ### Assignment for Task 7

# You have learned how to use different models and feature sets. For this assignment, use the knowledge that you learned to conduct experiments to answer the following questions:
# - Is including sensor data in the past a good idea to help improve model performance?
#   - Hint: Check precision, recall, and F1-score. 
# - Is the wind direction from the air quality monitoring station (i.e., feed 28) that near the pollution source a good feature to predict bad smell?
#   - Hint: Compare the performance of models with different feature sets and check the feature importance.
# 
# You need to tweak parameters in the code to conduct a pilot experiment to understand if wind direction is a good feature for predicting the presence of bad smell. Also, you need to inspect if including more data from the previous hours is helpful. Specifically, you need to do the experiment using the combinations of two models (Decision Tree and Random Forest) and four feature sets (as described below). So in total, there should be 8 situations (2 models and 4 feature sets) to compare.
# 
# The 1st feature set below has the H2S and SO2 data from the previous 1 hour:
# - `3.feed_28.H2S_PPM_pre_1h`
# - `3.feed_28.SO2_PPM_pre_1h`
# 
# The 2nd feature set below has the H2S, SO2, and wind data from the previous 1 hour:
# - `3.feed_28.H2S_PPM_pre_1h`
# - `3.feed_28.SO2_PPM_pre_1h`
# - `3.feed_28.SONICWD_DEG_sine_pre_1h`
# - `3.feed_28.SONICWD_DEG_cosine_pre_1h`
# 
# The 3rd feature set below has the H2S and SO2 data from the previous 2 hours:
# - `3.feed_28.H2S_PPM_pre_1h`
# - `3.feed_28.H2S_PPM_pre_2h`
# - `3.feed_28.SO2_PPM_pre_1h`
# - `3.feed_28.SO2_PPM_pre_2h`
# 
# The 4th feature set below has the H2S, SO2, and wind data from the previous 2 hours:
# - `3.feed_28.H2S_PPM_pre_1h`
# - `3.feed_28.SO2_PPM_pre_1h`
# - `3.feed_28.SONICWD_DEG_sine_pre_1h`
# - `3.feed_28.SONICWD_DEG_cosine_pre_1h`
# - `3.feed_28.H2S_PPM_pre_2h`
# - `3.feed_28.SO2_PPM_pre_2h`
# - `3.feed_28.SONICWD_DEG_sine_pre_2h`
# - `3.feed_28.SONICWD_DEG_cosine_pre_2h`

# In[41]:


def experiment(df_x, df_y):
    """
    Perform experiments and print the results.

    Parameters
    ----------
    df_x : pandas.DataFrame
        The data frame that contains all features.
    df_y : pandas.DataFrame
         The data frame that contains labels.
    """
    ###################################
    # Fill in your answer here
    print("None")
    ###################################


# In[42]:


experiment(df_x, df_y_40)


# You need to write the function above that can do similar things as the following from the answer.

# In[43]:


answer_experiment(df_x, df_y_40)


# Now, think about the following two questions that we asked you.
# - In your opinion, is including sensor data in the past a good idea to help improve model performance? 
# - In your opinion, is the wind direction from the air quality monitoring station (i.e., feed 28) that near the pollution source a good feature to predict bad smell?

# In[ ]:




