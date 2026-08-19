import pandas as pd
import numpy as np
import sys
import os

import argparse

sys.path.append(os.path.abspath('../'))
import Function.path as path
import Function.mouse as mouse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_name", required=True)
    args = parser.parse_args()

    node_name = args.node_name


    file_path = f'{path.data_path}/smoothed_corrected_raw.h5'
    df = pd.read_hdf(file_path)
    
    bodypart_data = {
    bodypart: (
        df.xs(bodypart, level="bodyparts", axis=1)
          .droplevel("scorer", axis=1)
          .loc[:, ["x", "y", "likelihood"]]
          .copy()
    )
    for bodypart in df.columns.get_level_values("bodyparts").unique()}

    processed_data = mouse.Kinematics(bodypart_data, node_name, use_parameter = False)
    processed_data.Run()
    
    np.savez(f'{path.data_path}/Parameters/{processed_data.node_name}_corrected.npz', **processed_data.parameters)    
    np.savez_compressed(f'{path.data_path}/filterRes/{processed_data.node_name}_corrected.npz', **processed_data.filterRes)
    np.savez_compressed(f'{path.data_path}/smoothRes/{processed_data.node_name}_corrected.npz', **processed_data.smoothRes) 
    
if __name__ == "__main__":
    main()