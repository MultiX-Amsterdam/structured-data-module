import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from util.util import compute_feature_importance
from util.util import train_and_evaluate


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
        # Copy the dataframe to avoid editing the original one.
        df = df.copy(deep=True)
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

    # Convert dataframe to float
    df = df.astype(float)

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

    # Convert dataframe to float
    df = df.astype(float)

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
    if n_hr == 0:
        df = df.astype(float)
        return df

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

    # Convert dataframe to float
    df = df.astype(float)

    return df


def answer_experiment(df_x, df_y):
    """
    This function is the answer of task 7.
    Perform experiments and print the results.

    Parameters
    ----------
    df_x : pandas.DataFrame
        The data frame that contains all features.
    df_y : pandas.DataFrame
         The data frame that contains labels.
    """
    wind = "3.feed_28.SONICWD_DEG"
    h2s = "3.feed_28.H2S_PPM"
    so2 = "3.feed_28.SO2_PPM"
    fs1 = [h2s + "_pre_1h", so2 + "_pre_1h"]
    w1 = [wind + "_sine_pre_1h", wind + "_cosine_pre_1h"]
    w2 = w1 + [wind + "_sine_pre_2h", wind + "_cosine_pre_2h"]
    fs1w = fs1 + w1
    fs2 = fs1 + [h2s + "_pre_2h", so2 + "_pre_2h"]
    fs2w = fs2 + w2
    feature_sets = [fs1, fs1w, fs2, fs2w]
    models = [DecisionTreeClassifier(), RandomForestClassifier()]
    for m in models:
        for fs in feature_sets:
            print("Use feature set %s" % (str(fs)))
            df_x_fs = df_x[fs]
            train_and_evaluate(m, df_x_fs, df_y, train_size=1440, test_size=168)
            compute_feature_importance(m, df_x_fs, df_y, scoring="f1")
            print("")
