import pandas as pd

import numpy as np

from ftplib import FTP, error_perm

import io
from io import BytesIO

from typing import Iterable, Optional, List

import sys

import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from kmodes.kprototypes import KPrototypes

import os

from kmedoids import KMedoids

import category_encoders as ce


def load_csvs_from_ftp_to_df(
    host: str="ftp.boxgrad.com",
    username: str="data@idxexchange.com",
    password: str="Real_estate123$",
    remote_dir: str="/raw",
    date_range: Iterable[int]=range(6, 12),
    prefix: str="california/CRMLSSold",
    year: int=2025,
    port: int=21,
    timeout: int=30,
    passive: bool=True,
    pandas_read_csv_kwargs: Optional[dict]=None,
    provided_local_dir: str=None,
) -> pd.DataFrame:
    """
    Load multiple CSV files from an FTP server into memory and return a single combined DataFrame.
    :param host: FTP host to connect to
    :param username: Username to authenticate with
    :param password: Password to authenticate with
    :param remote_dir: Remote directory to download from
    :param date_range: Range of dates to load
    :param prefix: Prefix to add to filenames
    :param year: Year to load
    :param port: Port to connect to
    :param timeout: Timeout to connect to
    :param passive: Passive mode
    :param pandas_read_csv_kwargs: Optional keyword arguments to pass to pandas.read_csv
    :param provided_local_dir: Local directory to download csv files
    :return: Combined DataFrame
    """
    # NEW: detect Google Colab and call colab_func
    if "google.colab" in sys.modules:
        try:
            from google.colab import auth
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
        except ModuleNotFoundError:
            raise ModuleNotFoundError()

        # Authenticate
        auth.authenticate_user()
        drive_service = build('drive', 'v3')

        # Find shared folder
        query = "sharedWithMe=true and name contains 'IDX_winter_26_ds45'"
        folder_id = drive_service.files().list(q=query, fields="files(id)").execute()["files"][0]["id"]

        # Find CSV files
        query = (
            f"'{folder_id}' in parents "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and name = 'dataset'"
        )
        dataset_folder_id = drive_service.files().list(
            q=query,
            fields="files(id)"
        ).execute()["files"][0]["id"]

        query = f"'{dataset_folder_id}' in parents and name contains 'CRMLS'"
        files = drive_service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()["files"]

        # Download and merge
        months = [str(year) + str(i) if len(str(i)) == 2 else str(year) + "0" + str(i) for i in date_range]
        dfs = []

        for file in files:
            if any(m in file["name"] for m in months):
                print(f"Loading {file['name']}...")
                request = drive_service.files().get_media(fileId=file["id"])
                buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(buffer, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                buffer.seek(0)
                dfs.append(pd.read_csv(buffer, low_memory=False))
        combined = pd.concat(dfs, ignore_index=True)

    elif provided_local_dir:
        combined = read_csvs(date_range=date_range, year=year, prefix=prefix)

    else:
        pandas_read_csv_kwargs = pandas_read_csv_kwargs or {}

        # Use a list of DataFrames, then concat once for speed.
        frames: List[pd.DataFrame] = []

        with FTP() as ftp:
            ftp.connect(host=host, port=port, timeout=timeout)
            ftp.login(user=username, passwd=password)
            ftp.set_pasv(passive)

            ftp.cwd(remote_dir)

            filenames = []

            for i in date_range:
                date = str(year) + str(i) if len(str(i)) == 2 else str(year) + "0" + str(i)
                filenames.append(prefix + date + ".csv")

            selected = list(filenames)

            for name in selected:
                bio = BytesIO()
                try:
                    ftp.retrbinary(f"RETR {name}", bio.write)
                except error_perm as e:
                    # Skip non-files / permission issues gracefully
                    continue

                bio.seek(0)

                # If server sends bytes, pandas can read BytesIO directly.
                df = pd.read_csv(bio, **pandas_read_csv_kwargs)

                frames.append(df)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined[(combined["PropertyType"] == "Residential") & (combined["PropertySubType"] == "SingleFamilyResidence")]
    return combined.drop(columns=["PropertyType", "PropertySubType"])


def read_csvs(
        date_range: Iterable[int],
        year: int=2025,
        prefix: str="california/CRMLSSold"
) -> pd.DataFrame:
    """
    Read all files within the range and filter corresponding rows
    :param date_range: range of files to read
    :param year: year of files to read
    :param prefix: prefix of the files
    :return: pandas DataFrame
    """

    dfs: List[pd.DataFrame] = []

    for i in date_range:
        date = str(year) + str(i) if len(str(i)) == 2 else str(year) + "0" + str(i)
        file_name = prefix + date + ".csv"
        dfs.append(pd.read_csv(file_name, low_memory=False))

    combined = pd.concat(dfs, ignore_index=True)

    filtered_df = combined[(combined["PropertyType"] == "Residential") & (combined["PropertySubType"] == "SingleFamilyResidence")]

    return filtered_df


def data_quality_summary(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a quality summary table
    :param df: raw data frame
    :return: pandas DataFrame
    """

    rows = []

    for col in df.columns:
        col_s = df[col]
        missing_pct = col_s.isna().mean() * 100
        missing_num = col_s.isna().sum()
        dtype = col_s.dtype
        nunique = col_s.nunique(dropna=True)

        rows.append({
            "column": col,
            "dtype": str(dtype),
            "n_unique": nunique,
            "missing_%": round(missing_pct, 2),
            "num_missing": missing_num,
        })

    return pd.DataFrame(rows).sort_values("missing_%", ascending=False)


def fill_na_with_mode(
        df: pd.DataFrame,
        target_col_list: list[str],
        reference_col: str,
        default_fill_val: str="Other"
) -> pd.DataFrame:
    """
    Fill missing values in target_col with mode according to reference_col
    :param df: raw data frame
    :param target_col_list: list of target columns need to be filled
    :param reference_col: reference column for the fill_val
    :param default_fill_val: default fill value
    :return: pandas DataFrame
    """
    for col in target_col_list:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(
            df.groupby(reference_col)[col]
            .transform(lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else default_fill_val)
        )

    return df


def create_flag(
        df: pd.DataFrame,
        target_col_list: list[str]
) -> pd.DataFrame:
    """
    Extract unique element, then using binary flag, fill the na with false
    :param df: raw data frame
    :param target_col_list: list of target categorical columns need to be filled
    :return: pandas DataFrame
    """
    for col in target_col_list:
        if col not in df.columns:
            continue
        unique_vals = (
            df[col]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
            .unique()
        )
        for val in unique_vals:
            df[val + "YN"] = df[col].str.contains(val)
            df[val + "YN"] = df[val + "YN"].fillna(False)
        # Drop redundant column
        df.drop([col], axis=1, inplace=True)
    return df


def impute_by_cluster(
    df: pd.DataFrame,
    target_col: str,
    cluster_col: str="cluster",
    fill_cat: str="mode",
    fill_num: str="median"
) -> pd.DataFrame:
    """
    Fill the na by cluster
    :param df: raw data frame
    :param cluster_col: column name of cluster column
    :param target_col: column name of target column
    :param fill_cat: method for categorical column
    :param fill_num: method for numerical column
    :return: pandas DataFrame
    """
    target_type = "num" if pd.api.types.is_numeric_dtype(df[target_col]) else "cat"
    for c in df[cluster_col].unique():
        mask = df[cluster_col] == c
        missing = mask & df[target_col].isna()

        if not missing.any():
            continue

        observed = df.loc[mask & df[target_col].notna(), target_col]

        # Categorical
        if target_type == "cat":
            if fill_cat == "mode" and not observed.empty:
                fill_value = observed.mode().iloc[0]
            else:
                fill_value = "Other"
        # Numerical
        elif target_type == "num":
            if observed.empty:
                fill_value = np.nan
            elif fill_num == "median":
                fill_value = observed.median()
            elif fill_num == "mean":
                fill_value = observed.mean()
            elif fill_num == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Unknown fill_num method: {fill_num}")
        else:
            raise ValueError("target_type must be 'cat' or 'num'")

        df.loc[missing, target_col] = fill_value

    return df


def knn_impute_latlon(
        df: pd.DataFrame,
        target_col_list: list[str],
        ref_col_List: list[str]=["Longitude", "Latitude"],
        metric="haversine",
        k: int=3,
        model=None,
        train_df_ref=None,
) -> (pd.DataFrame, NearestNeighbors):
    """
    KNN imputation using Haversine distance on (lat, lon) in radians.
    Numeric targets  -> mean of neighbors
    Categorical      -> mode of neighbors
    :param df: raw data frame
    :param target_col_list: list of target columns need to be filled
    :param ref_col_List: list of reference columns
    :param metric: distance metric
    :param k: number of nearest neighbors
    :param model: NearestNeighbors model
    :param train_df_ref: pandas DataFrame
    :return: pandas DataFrame, NearestNeighbors, pandas DataFrame
    """
    target_col_list = list(target_col_list)

    complete_coord_idx = df[ref_col_List].notna().any(axis=1)

    coords = np.radians(df.loc[complete_coord_idx, ref_col_List].astype(float).values)

    if model is None:
        # Copy train df for future reference
        train_df_ref = df.copy()
        # Fit KNN model
        model = NearestNeighbors(n_neighbors=k, metric=metric)
        model.fit(coords)

        global_fill = {}
        for col in target_col_list:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                global_fill[col] = df[col].mean()
            else:
                global_fill[col] = df[col].mode(dropna=True).iloc[0]

    if coords.shape[0] > 0:
        # Use KNN model to classify rows
        _, idx = model.kneighbors(coords)

        complete_rows = df.index[complete_coord_idx].to_numpy()

        for col in target_col_list:
            if col not in df.columns:
                continue
            is_num = pd.api.types.is_numeric_dtype(df[col])

            # only iterate rows where this column is missing
            miss_rows = complete_rows[df.loc[complete_coord_idx, col].isna().to_numpy()]
            if miss_rows.size == 0:
                continue

            # map complete_rows_row -> position in idx
            pos_map = {r: p for p, r in enumerate(complete_rows)}

            for r in miss_rows:
                p = pos_map[r]
                vals = train_df_ref[col].iloc[idx[p]]

                if is_num:
                    fill = vals.mean(dropna=True)
                else:
                    m = vals.mode(dropna=True)
                    fill = m.iloc[0] if len(m) else np.nan

                if pd.isna(fill):
                    fill = df[col].mean() if is_num else df[col].mode(dropna=True).iloc[0]

                df.at[r, col] = fill

        # rows without lat/lon -> global fallback
    no_ref = ~complete_coord_idx
    for col in target_col_list:
        if col not in df.columns:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[col])
        df.loc[no_ref & df[col].isna(), col] = df[col].mean() if is_num else df[col].mode(dropna=True).iloc[0]

    return df, model, train_df_ref


def get_cluster_through_k_means(
        df: pd.DataFrame,
        reference_col_list: list[str],
        num_cols: list[str] = None,
        cat_cols: list[str] = None,
        n_clusters: int = 25,
        random_state=42,
        model: KMeans = None,
        scaler_method: str = "robust",
        scaler=None,
) -> (pd.DataFrame, KMeans, object):
        """
        Using k-means to cluster data
        :param df: raw data frame
        :param reference_col_list: reference column used to compute distance
        :param num_cols: list of column names
        :param cat_cols: list of column names
        :param n_clusters: number of clusters
        :param random_state: random seed
        :param model: existed model to use
        :param scaler_method: method for scaling
        :param scaler: scaler to use
        :return: pandas DataFrame, KMeans, scaler
        """
        if num_cols is None and cat_cols is None:
            raise ValueError("num_cols and cat_cols cannot be None at the same time")
        if num_cols is None:
            num_cols = [x for x in reference_col_list if x not in cat_cols]

        X = df[num_cols].copy()

        if X.isna().any().any():
            raise (ValueError("NaN values not allowed"))

        # Scaler before the clustering
        X_c, scaler = normalize_df(X,
                                   num_col_list=num_cols,
                                   method=scaler_method,
                                   scaler=scaler,
                                   num_only=True)

        if model:
            # Testing data
            pass
        else:
            # Training data
            try:
                    model = KMeans(
                        n_clusters=n_clusters,
                        random_state=random_state
                    )
                    model.fit(X_c)
            except ValueError as e:
                print(e)

        labels = model.predict(X_c)
        df["cluster"] = labels
        return df, model, scaler


def get_cluster_through_k_medoids(
        df: pd.DataFrame,
        reference_col_list: list[str],
        num_cols: list[str] = None,
        cat_cols: list[str] = None,
        n_clusters: int = 25,
        random_state=42,
        model: KMedoids = None,
        scaler_method: str = "robust",
        scaler=None,
) -> (pd.DataFrame, KMedoids, object):
    """
    Using k-medoids to cluster data
    :param df: raw data frame
    :param reference_col_list: reference column used to compute distance
    :param num_cols: list of column names
    :param cat_cols: list of column names
    :param n_clusters: number of clusters
    :param random_state: random seed
    :param model: existed model to use
    :param scaler_method: method for scaling
    :param scaler: scaler to use
    :return: pandas DataFrame, KMedoids, scaler
    """
    if num_cols is None and cat_cols is None:
        raise ValueError("num_cols and cat_cols cannot be None at the same time")
    if num_cols is None:
        num_cols = [x for x in reference_col_list if x not in cat_cols]

    X = df[num_cols].copy()

    if X.isna().any().any():
        raise (ValueError("NaN values not allowed"))

    # Scaler before the clustering
    X_c, scaler = normalize_df(X,
                               num_col_list=num_cols,
                               method=scaler_method,
                               scaler=scaler,
                               num_only=True)

    if model:
        # Testing data
        pass
    else:
        # Training data
        try:

            model = KMedoids(
                n_clusters=n_clusters,
                metric="euclidean",
                method="fasterpam",
                random_state=random_state
            )

            model.fit(X_c)
        except ValueError as e:
            print(e)

    labels = model.predict(X_c)
    df["cluster"] = labels
    return df, model, scaler


def get_cluster_through_k_prototypes(
        df: pd.DataFrame,
        reference_col_list: list[str],
        num_cols: list[str]=None,
        cat_cols: list[str]=None,
        n_clusters: int=25,
        random_state=42,
        n_init: int=10,
        inits: list[str]=["Cao", "Huang", "random"],
        model: KPrototypes=None,
        scaler_method: str="robust",
        scaler=None,
) -> (pd.DataFrame, KPrototypes, object):
    """
    Using k-prototypes to cluster data
    :param df: raw data frame
    :param reference_col_list: reference column used to compute distance
    :param num_cols: list of column names
    :param cat_cols: list of column names
    :param n_clusters: number of clusters
    :param random_state: random seed
    :param n_init: number of init clusters
    :param inits: methods for initialization
    :param model: existed model to use
    :param scaler_method: method for scaling
    :param scaler: scaler to use
    :return: pandas DataFrame, KPrototypes, scaler
    """
    if num_cols is None and cat_cols is None:
        raise ValueError("num_cols and cat_cols cannot be None at the same time")
    if num_cols is None:
        num_cols = [x for x in reference_col_list if x not in cat_cols]
    elif cat_cols is None:
        cat_cols = [x for x in reference_col_list if x not in num_cols]

    X = df[num_cols + cat_cols].copy()

    if X.isna().any().any():
        raise(ValueError("NaN values not allowed"))

    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    # Scaler before the clustering
    X_c, scaler = normalize_df(X,
                       num_col_list=num_cols,
                       cat_col_list=cat_cols,
                       method=scaler_method,
                       scaler=scaler)

    if model:
        # Testing data
        pass
    else:
        # Training data
        for init in inits:
            try:
                model = KPrototypes(
                    n_clusters=n_clusters,
                    init=init,
                    n_init=n_init,
                    verbose=1,
                    max_iter=100,
                    random_state=random_state
                )
                model.fit(X_c, categorical=cat_idx)
                break
            except ValueError as e:
                print(e)

    labels = model.predict(X_c, categorical=cat_idx)
    df["cluster"] = labels
    return df, model, scaler


def fill_na_by_cluster(
        df: pd.DataFrame,
        target_col_list: list[str],
        reference_col_list: list[str],
        num_clusters: int=25,
        random_state=42,
        **kwargs
) -> (pd.DataFrame, object, object):
    """
    Clustering the data according to the reference column by method, then fill the na with fill_val
    :param df: raw data frame
    :param target_col_list: list of target columns need to be filled
    :param reference_col_list: list of reference column used to cluster
    :param num_clusters: number of clusters
    :param random_state: random seed
    :param kwargs:
        Optional keyword arguments:
        fill_method_num : str, default="median"
            Method to fill na for numerical data
        fill_method_num : str, default="mode"
            Method to fill na for categorical data
        method : str, default=k-prototypes
            Method to cluster
        model: object, default=None
            existed model to use
        scaler_method: str, default="robust
            method for scaling
        scaler: object, default=None
            scaler to use
    :return: (pandas DataFrame, model, scaler)
    """
    fill_method_num = kwargs.pop("fill_method_num", "median")
    fill_method_cat = kwargs.pop("fill_method_cat", "mode")
    method = kwargs.pop("method", "k-prototypes")
    model = kwargs.pop("model", None)
    scaler_method = kwargs.pop("scaler_method", "robust")
    scaler = kwargs.pop("scaler", None)

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs}")

    df_copy = df.copy(deep=True)

    num_cols = df_copy.select_dtypes(include="number").columns.tolist()
    num_cols = list(map(str, num_cols))
    reference_col_list = list(map(str, reference_col_list))
    num_reference_col_list = [col for col in num_cols if col in reference_col_list]

    if method == "k-prototypes":
        df_copy, model, scaler = get_cluster_through_k_prototypes(df=df_copy,
                                                          reference_col_list=reference_col_list,
                                                          num_cols=num_reference_col_list,
                                                          n_clusters=num_clusters,
                                                          random_state=random_state,
                                                          model=model,
                                                          scaler_method=scaler_method,
                                                          scaler=scaler)
    elif method == "k-means":
        df_copy, model, scaler = get_cluster_through_k_means(df=df_copy,
                                                             reference_col_list=reference_col_list,
                                                             num_cols=num_reference_col_list,
                                                             n_clusters=num_clusters,
                                                             random_state=random_state,
                                                             model=model,
                                                             scaler_method=scaler_method,
                                                             scaler=scaler)
    elif method == "k-medoids":
        df_copy, model, scaler = get_cluster_through_k_medoids(df=df_copy,
                                                             reference_col_list=reference_col_list,
                                                             num_cols=num_reference_col_list,
                                                             n_clusters=num_clusters,
                                                             random_state=random_state,
                                                             model=model,
                                                             scaler_method=scaler_method,
                                                             scaler=scaler)

    else:
        raise TypeError(f"Unexpected clustering method: {method}")

    for col in target_col_list:

        df_copy = impute_by_cluster(
            df_copy,
            cluster_col="cluster",
            target_col=col,
            fill_cat=fill_method_cat,
            fill_num=fill_method_num,
        )

    return df_copy.drop(["cluster"], axis=1), model, scaler


def drop_columns(
        df: pd.DataFrame,
        col_list: list[str],
) -> pd.DataFrame:
    """
    Drop columns from dataframe
    :param df: raw data frame
    :param col_list: list of columns to drop
    :return: pandas DataFrame
    """
    df = df.drop(col_list, axis=1, errors="ignore")

    return df


def convert_email_domains(
        df: pd.DataFrame,
        email_col_name: str="ListAgentEmail",
        domain_col_name: str="EmailDomain"
) -> pd.DataFrame:
    """
    Convert complete email into domain only
    :param df: raw data frame
    :param email_col_name: email column name need to be converted
    :param domain_col_name: domain column name need to be created
    :return: pandas DataFrame
    """
    if email_col_name in df.columns:
        df[domain_col_name] = df[email_col_name].str.split("@").str[1]

        df = df.drop(columns=[email_col_name])

    return df


def remove_extreme_rows(
        df: pd.DataFrame,
        price_col: str="ClosePrice",
        upper_bound_pct: float=0.995,
        lower_bound_pct: float=0.005,
) -> pd.DataFrame:
    """
    Remove erroneous or non-economic transactions, remove the top 0.5% and bottom 0.5% of ClosePrice
    :param df: raw data frame
    :param price_col: price column name
    :param upper_bound_pct: upper bound pct
    :param lower_bound_pct: lower bound pct
    :return: pandas DataFrame
    """
    if price_col not in df.columns:
        raise TypeError(f"Price column name not found: {price_col}")
    low = df[price_col].quantile(lower_bound_pct)
    high = df[price_col].quantile(upper_bound_pct)

    df = df[(df[price_col] >= low) & (df[price_col] <= high)]

    return df


def remove_by_positive_or_non_negative_constraint(
        df: pd.DataFrame,
        positive_col_list: list[str],
        non_negative_col_list: list[str],
) -> pd.DataFrame:
    """ 
    Remove non-positive and negative rows according to constraints
    :param df: raw data frame
    :param positive_col_list: list of columns with positive constraints
    :param non_negative_col_list: list of columns with non-negative constraints
    :return: pandas DataFrame
    """
    if positive_col_list is not None:
        for pos_col in positive_col_list:
            if pos_col not in df.columns:
                continue
            df = df[df[pos_col] > 0]
    if non_negative_col_list is not None:
        for col in non_negative_col_list:
            if col not in df.columns:
                continue
            df = df[df[col] >= 0]

    return df


def remove_by_location(
        df: pd.DataFrame,
        lat_min: float=32.5,
        lat_max: float=42.0,
        lon_min: float=-124.5,
        lon_max: float=-114.0
) -> pd.DataFrame:
    """
    Remove row that exceed the range for latitude, longitude
    :param df: raw data frame
    :param lat_min: minimum latitude
    :param lat_max: maximum latitude
    :param lon_min: minimum longitude
    :param lon_max: maximum longitude
    :return:
    """
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df = df[(df["Latitude"].between(lat_min, lat_max))
                            & (df["Longitude"].between(lon_min, lon_max))]
    else:
        raise TypeError("Latitude or Longitude not found")

    return df


def remove_by_missing_pct(
        df: pd.DataFrame,
        col_thresholds: dict[str, float]={},
        default_threshold: float=0.8
) -> (pd.DataFrame, list):
    """
    Remove columns with too much missing, according to thresholds, and return the col name to drop
    :param df: raw data frame
    :param col_thresholds: dict of column names to threshold
    :param default_threshold: default threshold to use for missing columns
    :return: (pandas DataFrame, list)
    """
    missing_frac = df.isna().mean()
    thresh_series = pd.Series(default_threshold, index=df.columns, dtype=float)
    for c, t in col_thresholds.items():
        if c in df.columns:
            if not (0 <= t <= 1):
                raise ValueError(f"Threshold for column '{c}' must be in [0, 1]")
            thresh_series.loc[c] = float(t)

    to_drop = missing_frac[missing_frac > thresh_series].index.tolist()
    df = df.drop(columns=to_drop)

    return df, to_drop


def save_file(
        df: pd.DataFrame,
        model_dict: dict=None,
        save_name: str="processed",
        data_type: str="train"
) -> str:
    """
    Save raw data frame to file
    :param df: raw data frame
    :param model_dict: dict of model needed to be saved
    :param save_name: file name
    :param data_type: data type (train, test)
    :return: None
    """
    i = 1
    if data_type == "train":
        new_save_name = save_name
        while os.path.exists(new_save_name):
            i += 1
            new_save_name = save_name + str(i)
        save_name = new_save_name
        os.mkdir(save_name)
        file_name = data_type + "_data"
        df.to_csv(save_name + "/" + file_name +  ".csv", index=False)
        for name, model in model_dict.items():
            with open(save_name + "/" + name + ".pkl", "wb") as f:
                pickle.dump(model, f)
    else:
        if os.path.exists(save_name):
            file_name = data_type + "_data"
            df.to_csv(save_name + "/" + file_name + ".csv", index=False)
    return save_name


def remove_duplicate(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate rows
    :param df: raw data frame
    :return: pandas DataFrame
    """
    return df.drop_duplicates()


def normalize_df(
        df: pd.DataFrame,
        num_col_list: list=None,
        cat_col_list: list=None,
        method:str="robust",
        scaler=None,
        num_only=False,
):
    """
    Normalize data frame
    :param df: raw data frame
    :param num_col_list: list of column names need to be normalized
    :param cat_col_list: list of categorical column names
    :param method: method for normalization
    :param scaler: scaler to use
    :param num_only: whether to only normalize numeric columns
    :return: normalized ndarray, scaler
    """
    if num_col_list is None:
        num_col_list = df.select_dtypes(include="number").columns.tolist()
    x_n = df[num_col_list].to_numpy()
    if num_only:
        pass
    else:
        if cat_col_list is None:
            cat_col_list = df.select_dtypes(include=["object", "category"]).columns.tolist()
        x_c = df[cat_col_list].astype(str).to_numpy()
    if scaler is None:
        if method == "robust":
            scaler = RobustScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"method not supported: {method}")

        scaler.fit(x_n)
    else:
        pass
    x_n_c = scaler.transform(x_n)
    X_c = x_n_c if num_only else np.hstack([x_n_c, x_c])
    return X_c, scaler


def recode_levels_df_apply(df,
                           tar_col="Levels",
                           dst_col="Levels_final"):
    """
    Recode target column
    :param df: raw data frame
    :param tar_col: target column name
    :param dst_col: final column name
    :return: pandas DataFrame
    """
    def recode(x):
        if pd.isna(x):
            return "Other"
        if "ThreeOrMore" in x:
            return "ThreeOrMore"
        elif "Two" in x:
            return "Two"
        elif "One" in x:
            return "One"
        elif "MultiSplit" in x:
            return "MultiSplit"
        else:
            return "Other"

    df[dst_col] = df[tar_col].apply(recode)

    return df.drop(columns=[tar_col])


def add_missing_indicators(
        df: pd.DataFrame,
        col_with_na: list=None,
) -> (pd.DataFrame, list):
    """
    Add missing indicators
    :param df: raw data frame
    :param col_with_na: list of column names
    :return: pandas DataFrame, list
    """
    if col_with_na is None:
        col_with_na = df.columns[df.isna().any()]

    missing_indicators = (
        df[col_with_na]
        .isna()
        .astype(int)
        .add_suffix("_missing")
    )

    return pd.concat([df, missing_indicators], axis=1), col_with_na


def pre_process(
        df: pd.DataFrame,
        **kwargs
) -> (pd.DataFrame, pd.DataFrame):
    """
    Pre-process raw data
    :param df: raw data frame
    :param kwargs:
        Optional keyword arguments:
        train_data : bool, default=True
            Whether dealing with training data or test data
        col_drop: list, default=[]
            List of columns to drop
        email_col: str, default="ListAgentEmail"
            Email column name
        new_email_cols: str, default="EmailDomain"
            new Email column name
        price_col: str, default="ClosePrice"
            Price column name
        upper_price_pct: float, default=0.995
            upper bound for price
        lower_price_pct: float, default=0.005
            lower bound for price
        positive_col_list: list, default=[]
            List of columns with positive constraints
        non_negative_col_list: list, default=[]
            List of columns with non-negative constraints
        lat_min: float, default=32.5
            Minimum latitude
        lat_max: float, default=42.0
            Maximum latitude
        long_min: float, default=-124.5
            Minimum longitude
        long_max: float, default=-114.0
            Maximum longitude
        col_thresholds: dict, default={}
            dict of column names and its missing pct thresholds
        default_threshold: float, default=0.8
            Default threshold to use for missing columns
        flag_col_list: list, default=["Levels", "Flooring"]
            List of columns need to do a binary-flag encoding
        yn_col_list: list, default=["AttachedGarageYN", "ViewYN", "NewConstructionYN", "PoolPrivateYN", "FireplaceYN"]
            list of binary column names
        col_fill_by_col_ref_mode_list: list, default=["CoListOfficeName", "MLSAreaMajor", "BuyerOfficeAOR", "BuyerOfficeName", "EmailDomain", "BuyerAgentAOR", "ListAgentAOR"]
            list of column names need to fill by refence column mode
        col_ref: str, default="City"
            Reference column name
        col_fill_by_knn_mode_list:list, default=["ElementarySchool", "MiddleOrJuniorSchool", "HighSchool", "HighSchoolDistrict"]
            list of column names need to fill by knn mode
        knn_col_ref_list: list, default=["Longitude", "Latitude"]
            list of column names use to conduct knn
        knn_k: int, default=3
            number of neighbors to use for knn
        knn_model: NearestNeighbors, default=None
            pre_trained NearestNeighbors model
        train_df_ref: pandas DataFrame, default=None
            Pandas DataFrame containing training data for knn reference
        num_clusters: int, default=10
            Number of clusters to form in k-prototypes
        clustering_method: str, default="k-means
            clustering method
        clustering_model: object, default=None
            pre_trained clustering model
        reference_col_list: list, default=None
            List of columns with reference columns used for clustering
        col_need_to_fill_na: list, default=None
            List of columns need to fill NaN by clustering
        random_state: float, default=42
            random seed
        scaler_method: str, default="robust
            method for scaling
        scaler: object, default=None
            scaler to use
        save_name: str, default="processed"
            Name of output folder
        save: bool, default=True
            Save processed data
        col_with_na: list, default=None
            List of columns with missing values
    :return:
    """
    train_data = kwargs.pop("train_data", True)
    col_drop_list = kwargs.pop("col_drop", [])
    email_col_name = kwargs.pop("email_col", "ListAgentEmail")
    domain_col_name = kwargs.pop("new_email_col", "EmailDomain")
    price_col = kwargs.pop("price_col", "ClosePrice")
    upper_price_pct = kwargs.pop("upper_price_pct", 0.995)
    lower_price_cpct = kwargs.pop("lower_price_cpct", 0.005)
    positive_col_list = kwargs.pop("positive_col_list", [])
    non_negative_col_list = kwargs.pop("non_negative_col_list", [])
    lat_min = kwargs.pop("lat_min", 32.5)
    lat_max = kwargs.pop("lat_max", 42.0)
    lon_min = kwargs.pop("lon_min", -124.5)
    lon_max = kwargs.pop("lon_max", -114.0)
    col_thresholds = kwargs.pop("col_thresholds", {})
    default_threshold = kwargs.pop("default_threshold", 0.8)
    flag_col_list = kwargs.pop("flag_col_list", ["Levels", "Flooring"])
    yn_col_list = kwargs.pop("yn_col_list", ["AttachedGarageYN",
                                             "ViewYN",
                                             "NewConstructionYN",
                                             "PoolPrivateYN",
                                             "FireplaceYN"])
    col_fill_by_col_ref_mode_list = kwargs.pop("col_fill_by_mode_list", ["CoListOfficeName",
                                                                        "MLSAreaMajor",
                                                                        "BuyerOfficeAOR",
                                                                        "BuyerOfficeName",
                                                                        "EmailDomain",
                                                                         "BuyerAgentAOR",
                                                                         "ListAgentAOR"])
    col_ref = kwargs.pop("col_ref", "City")
    col_fill_by_knn_mode_list = kwargs.pop("col_fill_by_knn_mode_list", ["ElementarySchool",
                                                                         "MiddleOrJuniorSchool",
                                                                         "HighSchool",
                                                                         "HighSchoolDistrict"])
    knn_col_ref_list = kwargs.pop("knn_col_ref_list", ["Longitude", "Latitude"])
    knn_k = kwargs.pop("knn_k", 3)
    knn_model = kwargs.pop("knn_model", None)
    train_df_ref = kwargs.pop("train_df_ref", None)
    num_clusters = kwargs.pop("num_clusters", 10)
    clustering_method = kwargs.pop("clustering_method", "k-means")
    clustering_model = kwargs.pop("clustering_model", None)
    random_state = kwargs.pop("random_state", 42)
    scaler_method = kwargs.pop("scaler_method", "robust")
    scaler = kwargs.pop("scaler", None)
    save_name = kwargs.pop("save_name", "processed")
    save = kwargs.pop("save", True)
    cols_with_na = kwargs.pop("cols_with_na", None)
    reference_col_list = kwargs.pop("reference_col_list", None)
    col_need_to_fill_na = kwargs.pop("col_need_to_fill_na", None)

    df_clean = df.copy()


    ### Remove columns
    if train_data:
        # Remove unrelated columns
        df_clean = drop_columns(df_clean,
                                col_list=col_drop_list)

        # Remove missing col by pct
        df_clean, col_drop_list = remove_by_missing_pct(df_clean,
                                         col_thresholds=col_thresholds,
                                         default_threshold=default_threshold)
        # Remove single-value columns

        single_value_cols = df_clean.columns[df_clean.nunique(dropna=True) <= 1].tolist()
        col_drop_list = col_drop_list + single_value_cols
        df_clean = df_clean.drop(columns=single_value_cols)
    else:
        if knn_model is None or clustering_model is None:
            raise ValueError("knn_model and clustering_model must be specified when processed test data")

        # Remove designated columns
        df_clean = drop_columns(df_clean,
                                col_list=col_drop_list)

    ### Remove rows

    # Remove duplicate rows
    df_clean = remove_duplicate(df_clean)

    # Convert email into email domain
    df_clean = convert_email_domains(df_clean,
                                     email_col_name=email_col_name,
                                     domain_col_name=domain_col_name)

    # Remove erroneous or non-economic transactions, remove the top 0.5% and bottom 0.5% of ClosePrice
    df_clean = remove_extreme_rows(df_clean,
                                   price_col=price_col,
                                   upper_bound_pct=upper_price_pct,
                                   lower_bound_pct=lower_price_cpct)

    # Remove non-positive and negative rows according to constraints
    df_clean = remove_by_positive_or_non_negative_constraint(df_clean,
                                                             positive_col_list=positive_col_list,
                                                             non_negative_col_list=non_negative_col_list)

    # Remove row that exceed the range for latitude, longitude
    df_clean = remove_by_location(df_clean,
                                  lat_min=lat_min,
                                  lat_max=lat_max,
                                  lon_min=lon_min,
                                  lon_max=lon_max)


    ### Handling missing

    # Add missing indicator
    df_clean, cols_with_na = add_missing_indicators(df_clean,
                                                    col_with_na=cols_with_na)

    #For "Levels", rencode it
    df_clean = recode_levels_df_apply(df_clean)

    # For "Flooring", using binary flag, fill the na with false
    df_clean = create_flag(df_clean,
                           target_col_list=flag_col_list)

    # For "CoListOfficeName", "MLSAreaMajor", "BuyerOfficeAOR", "BuyerOfficeName", "EmailDomain", fill with mode by city
    df_clean = fill_na_with_mode(df=df_clean,
                                 target_col_list=col_fill_by_col_ref_mode_list,
                                 reference_col=col_ref)

    # For school-related variables, Nearest-neighbor assignment through longitude and latitude
    df_clean, knn_model, train_df_ref = knn_impute_latlon(df_clean,
                                                          target_col_list=col_fill_by_knn_mode_list,
                                                          ref_col_List=knn_col_ref_list,
                                                          k=knn_k,
                                                          model=knn_model,
                                                          train_df_ref=train_df_ref)

    # For "YN" variables, fill the na with False
    df_clean[yn_col_list] = df_clean[yn_col_list].fillna(False)

    # For other variables, clustering according to all variables, then fill the na
    quality_summary_table = data_quality_summary(df_clean)
    col_need_to_fill_na = quality_summary_table[quality_summary_table["num_missing"] > 0]["column"] if col_need_to_fill_na is None else col_need_to_fill_na
    reference_col_list = quality_summary_table[quality_summary_table["num_missing"] == 0]["column"].drop(columns=price_col, errors="ignore") if reference_col_list is None else reference_col_list
    df_clean, clustering_model, scaler = fill_na_by_cluster(df=df_clean,
                                                            target_col_list=col_need_to_fill_na,
                                                            reference_col_list=reference_col_list,
                                                            method=clustering_method,
                                                            num_clusters=num_clusters,
                                                            random_state=random_state,
                                                            model=clustering_model,
                                                            scaler_method=scaler_method,
                                                            scaler=scaler)

    if save:
        save_name = save_file(df_clean,
                  save_name=save_name,
                  data_type="train" if train_data else "test",
                  model_dict={"knn_model": knn_model,
                              "clustering_model": clustering_model,
                              "scaler": scaler})


    return df_clean, knn_model, train_df_ref, reference_col_list, clustering_model, col_drop_list, scaler, cols_with_na, save_name


def mdape(y_true, y_pred):
    """
    Compute median absolute percentage error
    :param y_true: true data
    :param y_pred: predicted data
    :return: median absolute percentage error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.median(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100 if mask.any() else np.nan


def make_model_pipeline(
        model: object,
        num_cols: list=None,
        low_card_cols: list=None,
        high_card_cols: list=None,
        num_scaler: str="robust",
        smoothing: int=10,
        min_samples_leaf: int=20,
) -> object:
    """
    Create process pipeline
    :param model: model to be used
    :param num_cols: numerical columns
    :param low_card_cols: categorical columns with low card
    :param high_card_cols: categorical columns with high card
    :param num_scaler: numerical scaler
    :param smoothing: smoothing parameter
    :param min_samples_leaf: minimum number of samples leaf
    :return:
    """
    if low_card_cols is None or high_card_cols is None:
        raise ValueError("low_card_cols and high_card_cols cannot be None")
    num_sel = num_cols if num_cols is not None else selector(dtype_include=np.number)

    num_scaler = RobustScaler() if num_scaler == "robust" else StandardScaler()

    numeric_pipe = Pipeline(steps=[
        ("scaler", num_scaler)
    ])

    # One hot for low card col
    low_cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    # target for high card col
    high_cat_pipe = Pipeline([
        ("te", ce.TargetEncoder(
            smoothing=smoothing,
            min_samples_leaf=min_samples_leaf
        ))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_sel),
        ("low", low_cat_pipe, low_card_cols),
        ("high", high_cat_pipe, high_card_cols),
    ])

    return Pipeline(steps=[
            ("prep", preprocess),
            ("model", model)
        ])


def fit_predict(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        model: object,
        col_drop_list: list = None,
        card_threshold: int = 20,
        num_scaler: str="robust",
        smoothing: int=10,
        min_samples_leaf: int=20,
        log_transform: bool = False,
):
    """
    Fit model and predict data
    :param train_df: train data
    :param test_df: test data
    :param target_col: target column
    :param model: model to be used
    :param col_drop_list: columns to be dropped
    :param card_threshold: threshold between high card and low card
    :param num_scaler: numerical scaler
    :param smoothing: smoothing parameter
    :param min_samples_leaf: minimum number of samples leaf
    :param log_transform: log transform on target col or not
    :return:
    """
    col_drop_list = col_drop_list or []

    train_df = train_df.drop(columns=col_drop_list, errors="ignore")
    test_df = test_df.drop(columns=col_drop_list, errors="ignore")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    nunique = X_train[cat_cols].nunique(dropna=False)

    # split categorical col through card_threshold
    low_card_cols = nunique[(nunique <= card_threshold)].index.tolist()
    high_card_cols = nunique[(nunique >= card_threshold)].index.tolist()

    pipe = make_model_pipeline(model=model,
                               num_cols=num_cols,
                               low_card_cols=low_card_cols,
                               high_card_cols=high_card_cols,
                               num_scaler=num_scaler,
                               smoothing=smoothing,
                               min_samples_leaf=min_samples_leaf)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    if log_transform:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    return {
        "pipe": pipe,
        "model": model,
        "groups": {
            "low_card_ohe": low_card_cols,
            "high_card_te": high_card_cols
        },
        "r2": r2_score(y_test, y_pred),
        "mdape": mdape(y_test, y_pred),
    }