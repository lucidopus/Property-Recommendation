import json
import warnings
from io import StringIO
from datetime import date

import numpy as np
import pandas as pd
from pandas import DataFrame
from geopy import distance
from google.cloud import storage
from google.oauth2.service_account import Credentials
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from config import GOOGLE_JSON_PATH, BUCKET_NAME, FOCUS_RESALE_PATH, FOCUS_RENTAL_PATH

warnings.filterwarnings("ignore")

credentials = Credentials.from_service_account_file(f"{GOOGLE_JSON_PATH}")
client = storage.Client(credentials=credentials)


def get_blob_names(bucket: str = BUCKET_NAME, dirpath: str = FOCUS_RESALE_PATH):
    blobs = client.list_blobs(bucket, prefix=dirpath)
    blob_names = [blob.name for blob in blobs if blob.name.endswith(".csv")]
    return blob_names


def download_data_fragment(blob_name: str, bucket: str = BUCKET_NAME):
    try:
        blob = client.bucket(bucket).blob(blob_name)
        frag_data = blob.download_as_string().decode("utf-8")
        frag_df = pd.read_csv(StringIO(frag_data), sep="\001", lineterminator="\n")
    except Exception as e:
        error_info = {
            "file": blob.name,
            "error_class": str(e.__class__),
            "error_description": str(e),
        }
        print(error_info)
        frag_df = pd.DataFrame()
    return frag_df


def fetch_data_fragments(blob_names: list):
    with ThreadPoolExecutor(max_workers=4) as executor:
        fragments = executor.map(download_data_fragment, blob_names)
    fragment_list = [fragment for fragment in fragments]
    return fragment_list


def merge_data_fragments(fragments: list):
    merged = pd.concat(fragments, axis=0, ignore_index=True)
    return merged


def fetch_developer_name(json_string: str) -> str:
    try:
        json_string = json_string.replace('\\"', '"')
        json_string = json_string.replace("\\\\", "")
        json_string = json_string.replace("\\", '"')
        json_string = json_string[:-1]
        json_dict = json.loads(json_string)
        developer_name = json_dict["name"]
    except:
        developer_name = ""
    return developer_name


def fetch_unit_economics(json_string: str) -> str:
    try:
        json_string = json_string.replace('\\"', '"')
        json_string = json_string.replace("\\", '"')
        json_string = json_string[:-1]
        json_dict = json.loads(json_string)
        keys = ["unitCatType", "unitCatName", "unitSize", "bedroomCount", "highCost"]
        unit_economics = [
            [unit_obj["unit"][0][key] for key in keys] for unit_obj in json_dict
        ]
    except:
        unit_economics = []
    return unit_economics


def preprocess_rental_data(data: DataFrame):
    replace_unit_cat_names = {
        "Ind Floor": "Apartment",
        "Builder Floor": "Apartment",
        "Studio": "Apartment",
        "Penthouse": "Apartment",
        "Shop": "Office Space",
    }

    replace_bedroom_cat_names = {
        "1 Rk": "0.5",
        "1 RK": "0.5",
        "Studio": "0.75",
        "6+": "7",
    }

    object_type = [
        "projectid",
        "projectname",
        "city",
        "location",
        "sublocation",
        "unit_cat_name",
        "furnishing_status",
    ]

    numerical_type = [
        "latitude",
        "longitude",
        "unit_size",
        "unit_rent",
        "bedroom_count",
    ]

    column_order = [
        "projectid",
        "projectname",
        "city",
        "location",
        "sublocation",
        "unit_type",
        "latitude",
        "longitude",
        "unit_size",
        "bedroom_count",
        "unit_rent",
        "unit_cat_name",
        "unit_cat_type",
        "furnishing_status",
    ]
    data = data[
        (data["usertype"] == "EMPLOYEE") & (data["buildingtype"] == "Residential")
    ]

    focused_columns = [
        "propertyid",
        "projectname",
        "cityname",
        "micromarketname",
        "sublocalityname",
        "buildingtype",
        "buildinglatitude",
        "buildinglongitude",
        "atreainsqft",
        "number_of_rooms",
        "propertytype",
        "totalprice",
        "furnishing_status",
    ]
    rename_columns = [
        "projectid",
        "projectname",
        "city",
        "location",
        "sublocation",
        "unit_type",
        "latitude",
        "longitude",
        "unit_size",
        "bedroom_count",
        "unit_cat_name",
        "unit_rent",
        "furnishing_status",
    ]
    data = data.loc[:, focused_columns]
    data.columns = rename_columns

    data["unit_cat_type"] = data["unit_cat_name"].replace(replace_unit_cat_names)
    data["bedroom_count"] = data["bedroom_count"].replace(replace_bedroom_cat_names)

    object_type_dict = dict(map(lambda x: (x, object), object_type))
    numerical_type_dict = dict(map(lambda x: (x, float), numerical_type))
    data = data.astype({**object_type_dict, **numerical_type_dict})

    drop_indices = list(
        data[
            (data["latitude"] == 0)
            | (data["longitude"] == 0)
            | (data["unit_rent"] == 0)
            | (data["unit_size"] == 0)
        ].index
    )
    drop_indices += list(
        data[
            (data["latitude"].isna())
            | (data["longitude"].isna())
            | (data["unit_rent"].isna())
            | (data["unit_size"].isna())
            | (data["unit_cat_name"].isna())
            | (data["furnishing_status"].isna())
        ].index
    )
    drop_indices = list(set(drop_indices))
    data.drop(drop_indices, axis="index", inplace=True)
    data = data[column_order]
    data.reset_index(drop=True, inplace=True)
    return data


def preprocess_resale_data(data: DataFrame):
    replace_unit_cat_names = {
        "Ind Floor": "Apartment",
        "Builder Floor": "Apartment",
        "Studio": "Apartment",
        "Penthouse": "Apartment",
        "Plot": "Residential Plot",
        "Retail Shop": "Office Space",
    }
    focused_columns = [
        "projectid",
        "projectname",
        "developer",
        "city",
        "location",
        "sublocation",
        "status",
        "latitude",
        "longitude",
        "units",
    ]
    column_order = [
        "projectid",
        "developer_name",
        "projectname",
        "city",
        "location",
        "sublocation",
        "status",
        "unit_type",
        "unit_cat_name",
        "latitude",
        "longitude",
        "unit_size",
        "bedroom_count",
        "unit_cat_type",
        "unit_price",
    ]

    data = data.loc[:, focused_columns]
    data["unit_economics"] = data["units"].apply(lambda x: fetch_unit_economics(x))
    data["developer_name"] = data["developer"].apply(lambda x: fetch_developer_name(x))
    data.drop(["units", "developer"], axis="columns", inplace=True)
    explode_columns = ["unit_economics"]
    data = data.explode(explode_columns, ignore_index=True)
    unit_economics_df = data["unit_economics"].apply(pd.Series)
    unit_economics_df.columns = [
        "unit_type",
        "unit_cat_name",
        "unit_size",
        "bedroom_count",
        "unit_price",
    ]
    data.drop(["unit_economics"], axis="columns", inplace=True)
    data = pd.concat([data, unit_economics_df], axis=1)
    data["bedroom_count"] = data["bedroom_count"].astype("float")
    data.loc[
        ((data["bedroom_count"] == 0) & (data["unit_cat_name"] == "Studio")),
        "bedroom_count",
    ] = 0.75
    data.loc[
        ((data["bedroom_count"] == 0) & (data["unit_cat_name"] == "1 Rk")),
        "bedroom_count",
    ] = 0.5
    data["unit_cat_type"] = data["unit_cat_name"].replace(replace_unit_cat_names)
    object_type = [
        "projectid",
        "developer_name",
        "projectname",
        "city",
        "location",
        "sublocation",
        "status",
        "unit_type",
        "unit_cat_name",
        "unit_cat_type",
    ]
    numerical_type = [
        "latitude",
        "longitude",
        "unit_size",
        "bedroom_count",
        "unit_price",
    ]
    data.loc[:, object_type] = data.loc[:, object_type].astype("object")
    data.loc[:, numerical_type] = data.loc[:, numerical_type].astype("float")
    drop_indices = list(
        data[
            (data["latitude"] == 0)
            | (data["longitude"] == 0)
            | (data["unit_price"] == 0)
            | (data["unit_size"] == 0)
        ].index
    )
    drop_indices += list(
        data[
            (data["latitude"].isna())
            | (data["longitude"].isna())
            | (data["unit_price"].isna())
            | (data["unit_size"].isna())
            | (data["unit_cat_type"].isna())
        ].index
    )
    drop_indices = list(set(drop_indices))
    data.drop(drop_indices, axis="index", inplace=True)
    data = data[column_order]
    data.reset_index(drop=True, inplace=True)
    return data


def prepare_csv(listing_type: str):
    if listing_type == "Rental":
        blob_names = get_blob_names(BUCKET_NAME, FOCUS_RENTAL_PATH)
        fragments = fetch_data_fragments(blob_names=blob_names)
        data = merge_data_fragments(fragments=fragments)
        data = preprocess_rental_data(data)
    elif listing_type == "Resale":
        blob_names = get_blob_names(BUCKET_NAME, FOCUS_RESALE_PATH)
        fragments = fetch_data_fragments(blob_names=blob_names)
        data = merge_data_fragments(fragments=fragments)
        data = preprocess_resale_data(data)
    filename = (
        "_".join([listing_type, date.today().strftime(format="%d-%m-%Y")]) + ".csv"
    )
    data.to_csv(filename, index=False)
    return filename


def select_subset(data: DataFrame, city: str, unit_type: str):
    sub = data[(data["city"] == city) & (data["unit_type"] == unit_type)]
    sub.reset_index(drop=True, inplace=True)
    return sub


def get_unique_values(lead_data, focus_data, categorical):
    cat_cols_unique_data = []
    for col in categorical:
        temp = list(set(list(focus_data[col].unique()) + list(lead_data[col].unique())))
        cat_cols_unique_data.append(temp)
    return cat_cols_unique_data


def manhattan_custom(lead, focus):
    distance_arr = (
        (lead[:, 0] - focus[:, 0])
        + (-1) * (lead[:, 1] - focus[:, 1])
        + np.sum(np.abs((lead[:, 2:] - focus[:, 2:])), axis=1)
    )
    distance_arr = distance_arr.reshape(1, -1)
    return distance_arr


def inference_pipeline_rental(lead_params: dict, focus_data: DataFrame):
    replace_unit_cat_names = {
        "Ind Floor": "Apartment",
        "Builder Floor": "Apartment",
        "Studio": "Apartment",
        "Penthouse": "Apartment",
        "Shop": "Office Space",
    }
    replace_bedroom_cat_names = {
        "1 Rk": "0.5",
        "1 RK": "0.5",
        "Studio": "0.75",
        "6+": "7",
    }

    column_order = [
        "vector_distance",
        "similarity_score",
        "projectid",
        "projectname",
        "city",
        "unit_type",
        "unit_cat_name",
        "location",
        "sublocation",
        "latitude",
        "longitude",
        "distance",
        "unit_size",
        "bedroom_count",
        "unit_rent",
        "furnishing_status",
    ]

    lead_data = pd.DataFrame([lead_params])
    lead_data["distance"] = [0] * len(lead_data)
    lead_data["furnishing_status_match"] = [0] * len(lead_data)
    lead_data["unit_cat_type"] = lead_data["unit_cat_name"].replace(
        replace_unit_cat_names
    )
    lead_lat, lead_long, lead_rent, lead_size, lead_furnishing_status = lead_data[
        ["latitude", "longitude", "unit_rent", "unit_size", "furnishing_status"]
    ].to_numpy()[0]
    focus_data["distance"] = focus_data[["latitude", "longitude"]].apply(
        lambda x: distance.distance((lead_lat, lead_long), (x[0], x[1])).km, axis=1
    )
    focus_data["furnishing_status_match"] = focus_data["furnishing_status"].apply(
        lambda x: 0 if x == lead_furnishing_status else 1
    )
    concat_data = pd.concat([lead_data, focus_data], axis=0)
    concat_data = concat_data[
        (concat_data["distance"] <= 4.0)
        & (concat_data["unit_rent"] <= 1.25 * lead_rent)
        & (concat_data["unit_rent"] >= 0.5 * lead_rent)
        & (concat_data["unit_size"] <= 1.5 * lead_size)
        & (concat_data["unit_size"] >= 0.75 * lead_size)
    ]
    if len(concat_data) > 0:
        focus_subset = concat_data.iloc[1:, :]
        focus_subset.reset_index(drop=True, inplace=True)

        categorical_var = ["unit_cat_type"]
        (unit_cat_type_unique,) = get_unique_values(
            lead_data, focus_data, categorical_var
        )
        encoder = OneHotEncoder(
            categories=[unit_cat_type_unique], handle_unknown="ignore", sparse=False
        )
        unit_cat_type_unique = get_unique_values(lead_data, focus_data, categorical_var)

        encoder = OneHotEncoder(
            categories=unit_cat_type_unique, handle_unknown="ignore", sparse=False
        )
        encoded_data = encoder.fit_transform(concat_data[categorical_var])
        encoded_feature_names = encoder.get_feature_names_out(categorical_var)
        numerical_var = ["unit_size", "unit_rent", "bedroom_count", "distance"]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(concat_data[numerical_var])

        concat_arr = np.concatenate(
            (
                scaled_data,
                np.array(concat_data["furnishing_status_match"]).reshape(-1, 1),
                encoded_data,
            ),
            axis=1,
        )
        query_arr = concat_arr[0].reshape(1, concat_arr.shape[1])
        focus_arr = concat_arr[1:]

        distance_arr = manhattan_custom(query_arr, focus_arr)
        sorted_indices = np.argsort(distance_arr, axis=1)[0][:3]

        sim = lambda x: (1 / (1 + x)) * 100
        sim_scores = [
            sim(xi) if xi > 0 else 100.0 for xi in distance_arr[0][sorted_indices]
        ]

        result_data = pd.concat(
            [lead_data.iloc[[0], :], focus_subset.iloc[sorted_indices, :]], axis=0
        )
        result_data["vector_distance"] = np.append(
            np.array([np.nan]), distance_arr[0][sorted_indices]
        )
        result_data["similarity_rank"] = [np.nan] + list(
            range(1, len(sorted_indices) + 1)
        )
        result_data["similarity_score"] = [np.nan] + sim_scores
        result_data["type"] = np.array(["Lead"] + ["Focus"] * len(sorted_indices))
    else:
        result_data = lead_data
        result_data["type"] = np.array(["Lead"])
    result_data = result_data[column_order]
    return result_data


def inference_pipeline_resale(lead_params: dict, focus_data: DataFrame):
    replace_unit_cat_names = {
        "Ind Floor": "Apartment",
        "Builder Floor": "Apartment",
        "Studio": "Apartment",
        "Penthouse": "Apartment",
        "Plot": "Residential Plot",
        "Retail Shop": "Office Space",
    }

    column_order = [
        "similarity_score",
        "projectid",
        "projectname",
        "developer_name",
        "city",
        "unit_type",
        "unit_cat_name",
        "location",
        "sublocation",
        "status",
        "latitude",
        "longitude",
        "distance",
        "unit_size",
        "bedroom_count",
        "unit_price",
    ]

    lead_data = pd.DataFrame([lead_params])
    lead_data["distance"] = [0] * len(lead_data)
    lead_data["unit_cat_type"] = lead_data["unit_cat_name"].replace(
        replace_unit_cat_names
    )

    focus_data["distance"] = focus_data.loc[:, ["latitude", "longitude"]].apply(
        lambda x: distance.distance(
            (lead_params["latitude"], lead_params["longitude"]), (x[0], x[1])
        ).km,
        axis=1,
    )
    concat_data = pd.concat([lead_data, focus_data], axis=0)
    categorical_var = ["unit_cat_type"]
    (unit_cat_type_unique,) = get_unique_values(lead_data, focus_data, categorical_var)
    encoder = OneHotEncoder(
        categories=[unit_cat_type_unique], handle_unknown="ignore", sparse=False
    )
    encoded_data = encoder.fit_transform(concat_data[categorical_var])
    encoded_feature_names = encoder.get_feature_names_out(categorical_var)
    numerical_var = ["unit_size", "unit_price", "bedroom_count", "distance"]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(concat_data[numerical_var])
    concat_arr = np.concatenate((scaled_data, encoded_data), axis=1)
    query_arr = concat_arr[0].reshape(1, concat_arr.shape[1])
    focus_arr = concat_arr[1:]
    distance_arr = euclidean_distances(query_arr, focus_arr)
    sorted_indices = np.argsort(distance_arr, axis=1)[0][:3]
    sim = lambda x: (1 / (1 + x)) * 100
    sim_scores = [sim(xi) for xi in distance_arr[0][sorted_indices]]
    result_data = pd.concat(
        [lead_data.iloc[[0], :], focus_data.iloc[sorted_indices, :]], axis=0
    )
    result_data["similarity_score"] = [np.nan] + sim_scores
    result_data["similarity_rank"] = [np.nan] + list(range(1, 4, 1))
    result_data["type"] = np.array(["Lead"] + ["Focus"] * 3)
    result_data = result_data[column_order]
    return result_data
