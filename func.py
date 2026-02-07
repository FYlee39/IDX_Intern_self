import pandas as pd
import numpy as np
from ftplib import FTP, error_perm
from io import BytesIO
from typing import Iterable, Optional, Union, List
import sys
import io
from sklearn.neighbors import KNeighborsClassifier
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


def fill_col_by_knn(
        df : pd.DataFrame,
        target_col: str,
        reference_col_list: list[str],
        distance: str="radians",
        k=1,
        fallback_label="Other",
        model: KNeighborsClassifier=None,
) -> (pd.DataFrame, KNeighborsClassifier):
    """
    Fill the missing in target col through KNN
    :param df: dataframe
    :param target_col: column need to be filled
    :param reference_col_list: reference column used to compute distance
    :param distance: method for distance calculation
    :param k: number of neighbors to use
    :param fallback_label: label for data whose reference column are missing
    :param model: existed model to use
    :return: pandas DataFrame, KNeighborsClassifier
    """
    df_copy = df.copy()

    # Rows where target col is missing
    target_missing = df_copy[target_col].isna()

    # Rows with missing coordinates
    coord_missing = df_copy[reference_col_list].isna().any(axis=1)

    # target NA + any coord NA → "Other"
    fallback_idx = df_copy[target_missing & coord_missing].index
    df_copy.loc[fallback_idx, target_col] = fallback_label

    # Remaining rows needing fill
    fill_idx = df_copy[
        target_missing & ~coord_missing
    ].index

    if len(fill_idx) == 0:
        return df_copy

    # Known row with complete reference columns
    required_cols = reference_col_list + [target_col]

    known = df_copy[df_copy[required_cols].notna().all(axis=1)]

    # Coordinates (radians)
    if distance == "radians":
        X_known = np.radians(known[reference_col_list])
        y_known = known[target_col]

        X_missing = np.radians(df_copy.loc[fill_idx, reference_col_list])

    else:
        raise ValueError("distance not found")

    if model:
        # For testing data
        pass
    else:
        # For training data
        # KNN
        model = KNeighborsClassifier(
            n_neighbors=k,
            metric="haversine",
            weights="distance"
        )
        model.fit(X_known, y_known)

    # Predict
    df_copy.loc[fill_idx, target_col] = model.predict(X_missing)

    return df_copy, model


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
) -> pd.DataFrame:
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
    :return: pandas DataFrame
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

    if method == "knn":
        for col in target_col_list:
            df_copy = fill_col_by_knn(df=df_copy,
                                      target_col=col,
                                      reference_col_list=reference_col_list,
                                      distance=knn_distance,
                                      k=knn_k,
                                      model=model)
        return df_copy
    elif method == "k-prototypes":
        num_cols = df_copy.select_dtypes(include="number").columns.tolist()
        num_reference_col_list = list(set(num_cols) & set(reference_col_list))

        df_copy = get_cluster_through_k_prototypes(df=df_copy,
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

    return df_copy.drop(["cluster"], axis=1)


def pre_process(
        df: pd.DataFrame,
        positive_col_list: list[str]=None,
        non_negative_col_list: list[str]=None,
        missing_threshold: float=0.8,
        missing_col_list: list[str]=None,
        save=True,
        save_name=None,
        data_type: str="train",
        model_dict: dict=None,
        **kwargs
) -> pd.DataFrame:
    """
    Pre-process data
    :param df: raw data frame
    :param positive_col_list: col with positive constraint
    :param non_negative_col_list: col with non-negative constraint
    :param missing_threshold: remove the column with missing rate higher than this value
    :param missing_col_list: list of column names need to remove
    :param save: whether save the processed data to a new file
    :param save_name: name of the processed file
    :param data_type: str, default="train
    :param model_dict: existed model to use
    :param **kwargs:
        Optional keyword arguments:
        num_clusters : int, default=3
            number of clusters
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
        knn_k : int, default=1
            Number of nearest neighbors
        knn_distance : str, default="radians
            Method to compute distance for KNN
        random_state : random seed
    """
    num_clusters = kwargs.pop("num_clusters", 3)
    random_state = kwargs.pop("random_state", 42)

    df_clean = df.copy(deep=True)

    ### Remove unrelated columns

    # Remove PropertyType and PropertySubType
    df_clean = df_clean.drop(columns=["PropertyType", "PropertySubType"])

    # Remove agent and office name
    df_clean = df_clean.drop(columns=[
        "ListAgentFirstName",
        "ListAgentLastName",
        "ListAgentFullName",
        "CoListAgentFirstName",
        "CoListAgentLastName",
        "BuyerAgentFirstName",
        "BuyerAgentLastName",
        "CoBuyerAgentFirstName",
    ])

    # Remove id
    df_clean = df_clean.drop(columns=[
        "BuyerAgentMlsId",
        "ListingId",
        "ListingKey",
        "ListingKeyNumeric",
    ])

    # Extract Email Domains
    # Extract only the domain from email addresses (e.g., gmail.com, yahoo.com)
    # Extract email domain (part after @)
    df_clean["EmailDomain"] = df_clean["ListAgentEmail"].str.split("@").str[1]

    df_clean = df_clean.drop(columns=["ListAgentEmail"])


    ### Remove impossible/illogical rows

    # Remove erroneous or non-economic transactions, remove the top 0.5% and bottom 0.5% of ClosePrice
    low = df_clean["ClosePrice"].quantile(0.005)
    high = df_clean["ClosePrice"].quantile(0.995)

    df_clean = df_clean[(df_clean["ClosePrice"] >= low) & (df_clean["ClosePrice"] <= high)]

    # Remove non-positive and negative rows according to constraints
    if positive_col_list is not None:
        for pos_col in positive_col_list:
            df_clean = df_clean[df_clean[pos_col] > 0]
    if non_negative_col_list is not None:
        for col in non_negative_col_list:
            df_clean = df_clean[df_clean[col] >= 0]

    # remove row that exceed the range for latitude, longitude
    lat_min, lat_max = 32.5, 42.0
    lon_min, lon_max = -124.5, -114.0
    df_clean = df_clean[(df_clean["Latitude"].between(lat_min, lat_max))
                        & (df_clean["Longitude"].between(lon_min, lon_max))]

    # Convert the dtype of CloseDate
    df_clean["CloseDate"] = pd.to_datetime(df_clean["CloseDate"])

    ### Handle missing value

    # Given specific col to be removed
    if missing_col_list is not None:
        df_clean = df_clean.drop(columns=missing_col_list)
    else:
        df_clean = df_clean.loc[:, df_clean.isna().mean() < missing_threshold]

    # For "Levels", "Flooring", using binary flag, fill the na with false
    df_clean = create_flag(df_clean, ["Levels", "Flooring"])

    # For "CoListOfficeName", "MLSAreaMajor", "BuyerOfficeAOR", "BuyerOfficeName", "EmailDomain", fill with mode by city
    df_clean = fill_na_with_mode(df=df_clean,
                                 target_col_list=["CoListOfficeName", "MLSAreaMajor", "BuyerOfficeAOR",
                                                  "BuyerOfficeName", "EmailDomain"],
                                 reference_col="City")

    # For school-related variables, Nearest-neighbor assignment through longitude and latitude
    df_clean = fill_na_by_cluster(df=df_clean,
                                  target_col_list=["ElementarySchool", "MiddleOrJuniorSchool", "HighSchool",
                                                   "HighSchoolDistrict"],
                                  reference_col_list=["Longitude", "Latitude"],
                                  method="knn",
                                  model=model_dict.pop("knn"))

    # For "YN" variables, fill the na with False
    df_clean["AttachedGarageYN"] = df_clean["AttachedGarageYN"].fillna(False)
    df_clean["ViewYN"] = df_clean["ViewYN"].fillna(False)
    df_clean["NewConstructionYN"] = df_clean["NewConstructionYN"].fillna(False)
    df_clean["PoolPrivateYN"] = df_clean["PoolPrivateYN"].fillna(False)
    df_clean["FireplaceYN"] = df_clean["FireplaceYN"].fillna(False)

    # For "BuyerAgentAOR" and "ListAgentAOR", fill na with other
    df_clean["BuyerAgentAOR"] = df_clean["BuyerAgentAOR"].fillna("Other")
    df_clean["ListAgentAOR"] = df_clean["ListAgentAOR"].fillna("Other")

    # For other variables, clustering according to all variables, then fill the na
    quality_summary_table = data_quality_summary(df_clean)
    col_need_to_fill_na = quality_summary_table[quality_summary_table["num_missing"] > 0]["column"]
    reference_col_list = quality_summary_table[quality_summary_table["num_missing"] == 0]["column"]

    df_clean = fill_na_by_cluster(df=df_clean,
                                  target_col_list=col_need_to_fill_na,
                                  reference_col_list=reference_col_list,
                                  method="k-prototypes",
                                  num_clusters=num_clusters,
                                  random_state=random_state,
                                  model=model_dict.pop("k-prototypes")
                                  **kwargs)

    """
    Space for further processing
    """

    # Saving the processed data
    if save:
        save_name = save_name if save_name is not None else "processed"
        i = 1
        while os.path.exists(save_name + ".csv"):
            save_name = save_name + str(i)
            i += 1
        save_name = save_name + data_type + "_data"
        df_clean.to_csv(save_name + ".csv", index=False)

    return df_clean


if __name__ == '__main__':
    df = load_csvs_from_ftp_to_df(provided_local_dir="./")
    positive_col_list = ["BedroomsTotal",
                         "BathroomsTotalInteger",
                         "LotSizeAcres",
                         "LotSizeArea",
                         "LotSizeSquareFeet"]
    negative_col_list = ["ParkingTotal"]
    df_clean = pre_process(df,
                           positive_col_list=positive_col_list,
                           non_negative_col_list=negative_col_list,
                           missing_threshold=1.0)
    quality_summary_table = data_quality_summary(df_clean)
    print(quality_summary_table.sort_values(by=["missing_%"], ascending=False))