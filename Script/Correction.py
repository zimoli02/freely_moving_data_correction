import pandas as pd
import numpy as np
import sys
import os

import argparse

sys.path.append(os.path.abspath('../'))
import Script.INFORMATION as info
import Function.utility as util
import Function.mouse as mouse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_name", required=True)
    args = parser.parse_args()
    node_name = args.node_name

    # Turn .h5 file to pandas dataframe 
    bodypart_data = util.h5_to_pd(info.file_name)


    # Filtering and smoothing the positions
    processed_data = mouse.Kinematics(bodypart_data, node_name, info.video_frame_rate, use_parameter = False)
    processed_data.Run()
    
    # Save processed data
    util.save_processed_data(processed_data)
    
    
    # Save processed data to .h5 format
    util.
    
    
if __name__ == "__main__":
    main()