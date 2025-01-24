import pandas as pd
import numpy as np
import pytz
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.model_selection import cross_validate
import seaborn as sns
from sklearn.decomposition import PCA
from copy import deepcopy


def check_answer_df(df_result, df_answer, n=1):
    """
    This function checks if two output dataframes are the same.

    Parameters
    ----------
    df_result : pandas.DataFrame
        The result from the output of a function.
    df_answer: pandas.DataFrame
        The expected output of the function.
    n : int
        The numbering of the test case.
    """
    try:
        assert df_answer.equals(df_result)
        print("Test case %d passed." % n)
    except:
        print("Test case %d failed." % n)
        print("")
        print("Your output is:")
        print(df_result)
        print("")
        print("Expected output is:")
        print(df_answer)


def compute_feature_importance(model, df_x, df_y, scoring="f1"):
    """
    Compute feature importance of a model.

    Parameters
    ----------
    model : a sklearn model object
        The classifier model.
    df_x : pandas.DataFrame
        The dataframe with features.
    df_y : pandas.DataFrame
        The dataframe with labels.
    scoring : str
        A scoring function as documented below.
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """
    if model is None:
        model = RandomForestClassifier()
    print("Computer feature importance using", model)
    model.fit(df_x, df_y.squeeze())
    result = permutation_importance(model, df_x, df_y, n_repeats=10, random_state=0, scoring=scoring)
    feat_names = df_x.columns.copy()
    feat_ims = np.array(result.importances_mean)
    sorted_ims_idx = np.argsort(feat_ims)[::-1]
    feat_names = feat_names[sorted_ims_idx]
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    df = pd.DataFrame()
    df["feature_importance"] = feat_ims
    df["feature_name"] = feat_names
    print("=====================================================================")
    print(df)
    print("=====================================================================")


def scorer(model, x, y):
    """
    A customized scoring function to evaluate a classifier.

    Parameters
    ----------
    model : a sklearn model object
        The classifier model.
    x : pandas.DataFrame
        The feature matrix.
    y : pandas.Series
        The label vector.

    Returns
    -------
    dict of int or float
        A dictionary of evaluation metrics.
    """
    y_pred = model.predict(x)
    c = confusion_matrix(y, y_pred, labels=[0,1])
    p = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    a = accuracy_score(y, y_pred)
    return {"tn": c[0,0], "fp": c[0,1], "fn": c[1,0], "tp": c[1,1],
            "precision": p[0], "recall": p[1], "f1": p[2], "accuracy": a}


def train_and_evaluate(model, df_x, df_y, train_size=336, test_size=168):
    """
    Parameters
    ----------
    model : a sklearn model object
        The classifier model.
    df_x : pandas.DataFrame
        The dataframe with features.
    df_y : pandas.DataFrame
        The dataframe with labels.
    train_size : int
        Number of samples for training.
    test_size : int
        Number of samples for testing.
    """
    print("Use model", model)
    print("Perform cross-validation, please wait...")

    # Create time series splits for cross-validation.
    splits = []
    dataset_size = df_x.shape[0]
    for i in range(train_size, dataset_size, test_size):
        start = i - train_size
        end = i + test_size
        if (end >= dataset_size): break
        train_index = range(start, i)
        test_index = range(i, end)
        splits.append((list(train_index), list(test_index)))

    # Perform cross-validation.
    cv_res = cross_validate(model, df_x, df_y.squeeze(), cv=splits, scoring=scorer)

    # Print evaluation metrics.
    print("================================================")
    print("average f1-score:", round(np.mean(cv_res["test_f1"]), 2))
    print("average precision:", round(np.mean(cv_res["test_precision"]), 2))
    print("average recall:", round(np.mean(cv_res["test_recall"]), 2))
    print("average accuracy:", round(np.mean(cv_res["test_accuracy"]), 2))
    print("================================================")


def is_datetime_obj_tz_aware(dt):
    """
    Find if the datetime object is timezone aware.

    Parameters
    ----------
    dt : pandas.DatetimeIndex
        A datatime index object.
    """
    return dt.tz is not None


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


def plot_smell_by_day_and_hour(df):
    """
    Plot the average number of smell reports by the day of week and the hour of day.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed smell data.
    """
    # Copy the data frame to prevent editing the original one.
    df = df.copy(deep=True)

    # Convert timestamps to the local time in Pittsburgh.
    if is_datetime_obj_tz_aware(df.index):
        df.index = df.index.tz_convert(pytz.timezone("US/Eastern"))
    else:
        df.index = df.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))

    # Compute the day of week and the hour of day.
    df["day_of_week"] = df.index.dayofweek
    df["hour_of_day"] = df.index.hour

    # Plot the graph.
    y_l = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    df_pivot = pd.pivot_table(df, values="smell_value", index=["day_of_week"], columns=["hour_of_day"], aggfunc="mean")
    f, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df_pivot, annot=False, cmap="Blues", fmt="g", linewidths=0.1, yticklabels=y_l, ax=ax)


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


def get_pca_result(x, y, n=3):
    """
    Get the result of Principal Component Analysis.

    Parameters
    ----------
    x : pandas.DataFrame
        The features.
    y : pandas.DataFrame
        The labels.
    n : int
        Number of principal components.

    Returns
    -------
    df_pc : pandas.DataFrame
        A data frame with information about principal components.
    df_r : pandas.DataFrame
        A data frame with information about ratios of explained variances.
    """
    # Copy the data to prevent editing it.
    x, y = deepcopy(x), deepcopy(y)

    # Run the principal component analysis.
    pca = PCA(n_components=n)

    # Compute the eigenvectors, which are the principal components.
    pc = pca.fit_transform(x)
    columns = ["PC" + str(i) for i in range(1,1+n)]
    df_pc = pd.DataFrame(data=pc, columns=columns)

    # Set the label for plotting.
    df_pc["y"] = y.astype(str)

    # Set the marker size for plotting.
    df_pc["size"] = 15

    # Get eigenvalues (i.e., variances explained by principal components).
    r = np.round(pca.explained_variance_ratio_, 3)
    df_r = pd.DataFrame({"var":r, "pc":columns})
    return df_pc, df_r
