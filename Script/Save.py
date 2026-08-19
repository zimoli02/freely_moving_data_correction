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
    parser.add_argument(
        "--node_names",
        nargs="+",
        required=True,
        help="List of node names to process",
    )
    args = parser.parse_args()
    node_names = args.node_names
    print(f'Transferring smoothed data for the following nodes: {node_names}', flush=True)

    
    # Save processed position to .h5 format
    util.save_smoothed_data_to_h5(node_names)
    
    
    # overlay processed position to original video
    
    
    
if __name__ == "__main__":
    main()