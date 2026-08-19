import os 
import pandas as pd
import numpy as np

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))
import Script.INFORMATION as info

def h5_to_pd(file_name):
    df = pd.read_hdf(f'{info.raw_data_path}/{file_name}')
    names = list(df.columns.names)
    names_before_bodyparts = names[:names.index("bodyparts")]
    bodypart_data = {
        bodypart: (
            df.xs(bodypart, level="bodyparts", axis=1)
            .droplevel(names_before_bodyparts, axis=1)
            .loc[:, ["x", "y", "likelihood"]]
            .copy()
        )
        for bodypart in df.columns.get_level_values("bodyparts").unique()
    }
    return bodypart_data

def save_processed_data(processed_data):
    np.savez(f'{info.parameter_path}/Parameters/{processed_data.node_name}.npz', **processed_data.parameters)    
    np.savez_compressed(f'{info.data_path}/filterRes/{processed_data.node_name}.npz', **processed_data.filterRes)
    np.savez_compressed(f'{info.data_path}/smoothRes/{processed_data.node_name}.npz', **processed_data.smoothRes) 


def save_smoothed_data_to_h5(file_name, smoothRes):

    # Load the original DLC dataframe.
    corrected_df = pd.read_hdf(info.raw_data_path).copy()
    if not isinstance(corrected_df.columns, pd.MultiIndex):
        raise ValueError(
            "The original H5 file does not have DLC-style MultiIndex columns."
        )

    column_names = list(corrected_df.columns.names)
    try:
        bodypart_level = column_names.index("bodyparts")
        coordinate_level = column_names.index("coords")
    except ValueError:
        # Standard single-animal DLC format:
        # scorer -> bodyparts -> coords
        if corrected_df.columns.nlevels == 3:
            bodypart_level = 1
            coordinate_level = 2
        elif corrected_df.columns.nlevels == 4:
            bodypart_level = 2
            coordinate_level = 3
        else:
            raise ValueError(
                "Could not identify the bodypart and coordinate column levels. "
                f"Column levels are: {column_names}"
            )

    n_frames = len(corrected_df)

    available_bodyparts = set(
        corrected_df.columns.get_level_values(bodypart_level)
    )
    
    
    corrected_data = {
        node_name: {
            "x": np.load(f"{path.data_path}/smoothRes/{node_name}.npz")['xnN'][0][0],
            "y": np.load(f"{path.data_path}/smoothRes/{node_name}.npz")['xnN'][3][0]
        }
        for node_name in bodypart_data.keys()
    }

    for node_name, node_data in corrected_data.items():
        if node_name not in available_bodyparts:
            print(
                f"Warning: {node_name!r} is not present in the original "
                "DLC file and was skipped."
            )
            continue

        x = np.asarray(node_data["x"], dtype=float)
        y = np.asarray(node_data["y"], dtype=float)

        if len(x) != n_frames or len(y) != n_frames:
            raise ValueError(
                f"{node_name!r} has {len(x)} x-values and {len(y)} y-values, "
                f"but the DLC file has {n_frames} frames."
            )

        # Find the full MultiIndex columns corresponding to this node's x and y.
        x_columns = [
            column
            for column in corrected_df.columns
            if (
                column[bodypart_level] == node_name
                and column[coordinate_level] == "x"
            )
        ]

        y_columns = [
            column
            for column in corrected_df.columns
            if (
                column[bodypart_level] == node_name
                and column[coordinate_level] == "y"
            )
        ]

        if not x_columns or not y_columns:
            raise KeyError(
                f"Could not find x/y columns for body part {node_name!r}"
            )

        # Usually there is one x column and one y column per body part.
        for column in x_columns:
            corrected_df.loc[:, column] = x

        for column in y_columns:
            corrected_df.loc[:, column] = y

        # Likelihood is intentionally preserved from the original file.

    output_h5_path = path.processed_data_path
    os.makedirs(output_h5_path, exist_ok=True)

    corrected_df.to_hdf(
        f'{output_h5_path}/{file_name}.h5',
        key="df_with_missing",
        mode="w",
        format="table",
    )

    print(f"Saved corrected DLC file to: {output_h5_path}", flush = True)
