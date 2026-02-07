import pandas as pd
import numpy as np
from ftplib import FTP, error_perm
from io import BytesIO
from typing import Iterable, Optional, Union, List
import sys
import io
from sklearn.pipeline import Pipeline
import pickle

from fontTools.misc.cython import returns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from kmodes.kprototypes import KPrototypes
import os


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
        return read_csvs(date_range=date_range, year=year, prefix=prefix)

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
    return combined[(combined["PropertyType"] == "Residential") & (combined["PropertySubType"] == "SingleFamilyResidence")]


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
    df_copy = df.copy()

    for col in target_col_list:
        df_copy[col] = df_copy[col].fillna(
            df_copy.groupby(reference_col)[col]
            .transform(lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else default_fill_val)
        )

    return df_copy


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
    df_copy = df.copy()
    for col in target_col_list:
        unique_vals = (
            df_copy[col]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
            .unique()
        )
        for val in unique_vals:
            df_copy[val + "YN"] = df_copy[col].str.contains(val)
            df_copy[val + "YN"] = df_copy[val + "YN"].fillna(False)
        # Drop redundant column
        df_copy.drop([col], axis=1, inplace=True)
    return df_copy


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
    :return: pandas DataFrame
    """
    target_col_list = list(target_col_list)

    complete_coord_idx = df[ref_col_List].notna().any(axis=1)

    coords = np.radians(df.loc[complete_coord_idx, ref_col_List].astype(float).values)

    if model is None:
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
                vals = df[col].iloc[idx[p]]

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
        is_num = pd.api.types.is_numeric_dtype(df[col])
        df.loc[no_ref & df[col].isna(), col] = df[col].mean() if is_num else df[col].mode(dropna=True).iloc[0]

    return df, model


def get_cluster_through_k_prototypes(
        df: pd.DataFrame,
        reference_col_list: list[str],
        num_cols: list[str]=None,
        cat_cols: list[str]=None,
        n_clusters: int=25,
        random_state=42,
        n_init: int=10,
        inits: list[str]=["Cao", "Huang", "random"],
        model: KPrototypes=None
) -> (pd.DataFrame, KNeighborsClassifier):
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
    :return: pandas DataFrame, KNeighborsClassifier
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

    X_np = X.to_numpy()

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

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
                    verbose=0,
                    random_state=random_state
                )
                break
            except ValueError as e:
                print(e)

    labels = model.fit_predict(X_np, categorical=cat_idx)
    df["cluster"] = labels
    return df, model


def fill_na_by_cluster(
        df: pd.DataFrame,
        target_col_list: list[str],
        reference_col_list: list[str],
        num_clusters: int=25,
        random_state=42,
        **kwargs
) -> (pd.DataFrame, object):
    """
    Clustering the data according to the reference column by method, then fill the na with fill_val
    :param df: raw data frame
    :param target_col_list: list of target columns need to be filled
    :param reference_col_list: list of reference column used to cluster
    :param num_clusters: number of clusters
    :param random_state: random seed
    :param **kwargs:
        Optional keyword arguments:
        distance : function, default=compute_gower
            Method to compute distance
        num_cluster : int, default=3
            Number of clusters iterations.
        fill_method_num : str, default="median"
            Method to fill na for numerical data
        fill_method_num : str, default="mode"
            Method to fill na for categorical data
        method : str, default=None
            Method to cluster
        knn_k: int, default=1
            Number of nearest neighbors
        knn_distance: str, default="radians
            Method to compute distance for KNN
        model: object, default=None
            existed model to use
    :return: (pandas DataFrame, model)
    """
    fill_method_num = kwargs.pop("fill_method_num", "median")
    fill_method_cat = kwargs.pop("fill_method_cat", "mode")
    method = kwargs.pop("method", None)
    knn_k = kwargs.pop("knn_k", 1)
    knn_distance = kwargs.pop("knn_distance", "radians")
    model = kwargs.pop("model", None)

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs}")

    df_copy = df.copy(deep=True)

    if method == "k-prototypes":
        num_cols = df_copy.select_dtypes(include="number").columns.tolist()
        num_reference_col_list = list(set(num_cols) & set(reference_col_list))

        df_copy, model = get_cluster_through_k_prototypes(df=df_copy,
                                                   reference_col_list=reference_col_list,
                                                   num_cols=num_reference_col_list,
                                                   n_clusters=num_clusters,
                                                   random_state=random_state,
                                                   model=model)
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

    return df_copy.drop(["cluster"], axis=1), model


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
    else:
        print(email_col_name + "not found")

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
    :return:
    """
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
            df = df[df[pos_col] > 0]
    if non_negative_col_list is not None:
        for col in non_negative_col_list:
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
        raise TypeError("Latitude and Longitude not found")

    return df


def remove_by_missing_pct(
        df: pd.DataFrame,
        col_thresholds: dict[str, float]={},
        default_threshold: float=0.8
) -> pd.DataFrame:
    """
    Remove columns with too much missing, according to thresholds
    :param df: raw data frame
    :param col_thresholds: dict of column names to threshold
    :param default_threshold: default threshold to use for missing columns
    :return: pandas DataFrame
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

    return df


def save_file(
        df: pd.DataFrame,
        model_dict: dict=None,
        save_name: str="processed",
        data_type: str="train"
) -> None:
    """
    Save raw data frame to file
    :param df: raw data frame
    :param model_dict: dict of model needed to be saved
    :param save_name: file name
    :param data_type: data type (train, test)
    :return: None
    """
    i = 1
    while os.path.exists(save_name):
        save_name = save_name + str(i)
        i += 1
    os.mkdir(save_name)
    file_name = data_type + "_data"
    df.to_csv(save_name + "/" + file_name +  ".csv", index=False)
    for name, model in model_dict.items():
        with open(save_name + "/" + name + ".pkl", "wb") as f:
            pickle.dump(model, f)


def pre_process(
        df: pd.DataFrame,
        **kwargs

) -> (pd.DataFrame, pd.DataFrame):
    """
    Pre-process raw data
    :param df: raw data frame
    :return: ()
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
    num_clusters = kwargs.pop("num_clusters", 10)
    k_prototypes_model = kwargs.pop("k_prototypes_model", None)
    random_state = kwargs.pop("random_state", 42)
    save_name = kwargs.pop("save_name", "processed")
    save = kwargs.pop("save", True)

    df_clean = df.copy()


    ### Remove columns
    if train_data:
        # Remove unrelated columns
        df_clean = drop_columns(df_clean,
                                col_list=col_drop_list)

        # Remove missing col by pct
        df_clean = remove_by_missing_pct(df_clean,
                                         col_thresholds=col_thresholds,
                                         default_threshold=default_threshold)
    else:
        if knn_model is None or k_prototypes_model is None:
            raise ValueError("knn_model and k_prototypes_model must be specified when processed test data")

        # Remove designated columns
        df_clean = drop_columns(df_clean,
                                col_list=col_drop_list)

    ### Remove rows

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

    # For "Levels", "Flooring", using binary flag, fill the na with false
    df_clean = create_flag(df_clean,
                           target_col_list=flag_col_list)

    # For "CoListOfficeName", "MLSAreaMajor", "BuyerOfficeAOR", "BuyerOfficeName", "EmailDomain", fill with mode by city
    df_clean = fill_na_with_mode(df=df_clean,
                                 target_col_list=col_fill_by_col_ref_mode_list,
                                 reference_col=col_ref)

    # For school-related variables, Nearest-neighbor assignment through longitude and latitude
    df_clean, knn_model = knn_impute_latlon(df_clean,
                                            target_col_list=col_fill_by_knn_mode_list,
                                            ref_col_List=knn_col_ref_list,
                                            k=knn_k,
                                            model=knn_model)

    # For "YN" variables, fill the na with False
    df_clean[yn_col_list] = df_clean[yn_col_list].fillna(False)

    # For other variables, clustering according to all variables, then fill the na
    quality_summary_table = data_quality_summary(df_clean)
    col_need_to_fill_na = quality_summary_table[quality_summary_table["num_missing"] > 0]["column"]
    reference_col_list = quality_summary_table[quality_summary_table["num_missing"] == 0]["column"]
    df_clean, k_prototypes_model = fill_na_by_cluster(df=df_clean,
                                  target_col_list=col_need_to_fill_na,
                                  reference_col_list=reference_col_list,
                                  method="k-prototypes",
                                  num_clusters=num_clusters,
                                  random_state=random_state,
                                  model=k_prototypes_model)

    if save:
        save_file(df_clean,
                  save_name=save_name,
                  data_type="train" if train_data else "test")


    return df_clean, knn_model, k_prototypes_model

if __name__ == '__main__':
    df = load_csvs_from_ftp_to_df(provided_local_dir="./")
    positive_col_list = ["BedroomsTotal",
                         "BathroomsTotalInteger",
                         "LotSizeAcres",
                         "LotSizeArea",
                         "LotSizeSquareFeet"]
    non_negative_col_list = ["ParkingTotal"]
    df_clean = pre_process(df,
                           positive_col_list=positive_col_list,
                           non_negative_col_list=non_negative_col_list,
                           knn_k=3)
    quality_summary_table = data_quality_summary(df_clean)
    print(quality_summary_table.sort_values(by=["missing_%"], ascending=False))