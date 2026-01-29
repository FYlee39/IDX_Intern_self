import pandas as pd
import numpy as np
from ftplib import FTP, error_perm
from io import BytesIO
from typing import Iterable, Optional, Union, List
import sys
import io


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
        dtype = col_s.dtype
        nunique = col_s.nunique(dropna=True)

        # Default outlier pct
        outlier_pct = np.nan

        # If the dtype is numeric, check the outlier rate
        if pd.api.types.is_numeric_dtype(col_s):
            col_s = col_s.dropna()
            if len(col_s) > 0:
                q1 = col_s.quantile(0.25)
                q3 = col_s.quantile(0.75)
                iqr = q3 - q1

                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_pct = ((col_s < lower) | (col_s > upper)).mean() * 100
                else:
                    outlier_pct = 0.0  # constant column

        rows.append({
            "column": col,
            "dtype": str(dtype),
            "n_unique": nunique,
            "missing_%": round(missing_pct, 2),
            "outlier_%": round(outlier_pct, 2) if not np.isnan(outlier_pct) else np.nan
        })

    return pd.DataFrame(rows).sort_values("missing_%", ascending=False)

def pre_process(
        df: pd.DataFrame,
        columns=None,
        categorical_cols=None,
) -> pd.DataFrame:
    """
    Pre-process data
    :param df: raw data frame
    :param columns: columns to preprocess
    :param categorical_cols: categorical columns
    :return: pandas DataFrame
    """

    # Remove PropertyType and PropertySubType
    df = df.drop(columns=["PropertyType", "PropertySubType"])

    # Remove entries with non-positive close price
    df = df[df["ClosePrice"] > 0]

    # Remove columns with 0 non-null value
    df = df.dropna(axis=1, how="all")

    # Convert the dtype of CloseDate
    df["CloseDate"] = pd.to_datetime(df["CloseDate"])

    # Remove agent and office name
    df = df.drop(columns=[
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
    df = df.drop(columns=[
        "BuyerAgentMlsId",
        "ListingId",
        "ListingKey",
        "ListingKeyNumeric",
    ])

    # Extract Email Domains
    # Extract only the domain from email addresses (e.g., gmail.com, yahoo.com)
    # Extract email domain (part after @)
    df["EmailDomain"] = df["ListAgentEmail"].str.split("@").str[1]

    df = df.drop(columns=[
        "ListAgentEmail"
    ])

    # Remove identical columns
    df = df.loc[:, df.nunique(dropna=True) > 1]

    if columns is None:
        columns = list(df.columns)
    df_clean = df[columns].copy()
    # Infer types if not provided
    if categorical_cols is None:
        numeric_cols = list(df_clean.select_dtypes(include=["number"]).columns)
        categorical_cols = [c for c in columns if c not in numeric_cols]
    else:
        categorical_cols = list(categorical_cols)
        numeric_cols = [c for c in columns if c not in categorical_cols]

    df_clean[categorical_cols] = df_clean[categorical_cols].astype("category")

    # Remove erroneous or non-economic transactions, remove the top 0.5% and bottom 0.5% of ClosePricevalues
    low = df_clean["ClosePrice"].quantile(0.005)
    high = df_clean["ClosePrice"].quantile(0.995)

    df_clean = df_clean[(df_clean["ClosePrice"] >= low) & (df_clean["ClosePrice"] <= high)]

    """
    Space for further processing
    """

    return df_clean


if __name__ == '__main__':
    df = load_csvs_from_ftp_to_df(provided_local_dir="./")
    print(df.head())
