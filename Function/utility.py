import os 
import pandas as pd
import numpy as np
import cv2

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))
import Script.INFORMATION as info

def h5_to_pd():
    df = pd.read_hdf(f'{info.raw_data_path}/{info.file_name}')
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
    os.makedirs(info.parameter_path, exist_ok=True)
    os.makedirs(f'{info.data_path}/filterRes', exist_ok=True)
    os.makedirs(f'{info.data_path}/smoothRes', exist_ok=True)
    
    np.savez(f'{info.parameter_path}/{processed_data.node_name}.npz', **processed_data.parameters)    
    np.savez_compressed(f'{info.data_path}/filterRes/{processed_data.node_name}.npz', **processed_data.filterRes)
    np.savez_compressed(f'{info.data_path}/smoothRes/{processed_data.node_name}.npz', **processed_data.smoothRes) 
    print('Finish saving processed data\n', flush = True)

def save_smoothed_data_to_h5(node_names):
    # Load the original DLC dataframe.
    corrected_df = pd.read_hdf(f'{info.raw_data_path}/{info.file_name}').copy()
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
            "x": np.load(f"{info.data_path}/smoothRes/{node_name}.npz")['xnN'][0][0],
            "y": np.load(f"{info.data_path}/smoothRes/{node_name}.npz")['xnN'][3][0]
        }
        for node_name in node_names
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

    output_h5_path = info.processed_data_path
    os.makedirs(output_h5_path, exist_ok=True)

    corrected_df.to_hdf(
        f'{output_h5_path}/correct.h5',
        key="df_with_missing",
        mode="w",
        format="table",
    )

    print(f"Saved corrected DLC file to: {output_h5_path}/correct.h5", flush = True)

def overlay_points_on_video(node_names, point_radius = 7):
    def draw_title(frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        text_x = (frame_width - text_width) // 2
        text_y = (header_height + text_height) // 2

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_positions(frame, positions, frame_index):
        for node_name in node_names:
            x = positions[node_name]["x"][frame_index]
            y = positions[node_name]["y"][frame_index]

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            x = int(round(x))
            y = int(round(y))

            if not (
                0 <= x < frame_width
                and 0 <= y < frame_height
            ):
                continue

            point = (x, y + header_height)

            # Filled node-specific color
            cv2.circle(
                frame,
                point,
                point_radius,
                node_colors[node_name],
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            # Black outline
            cv2.circle(
                frame,
                point,
                point_radius,
                (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            
    # Load tracking DataFrames
    raw_df = pd.read_hdf(f'{info.raw_data_path}/{info.file_name}')
    processed_df = pd.read_hdf(f'{info.raw_data_path}/correct.h5')

    if info.processed_video_path is None:
        if not hasattr(info, "processed_video_path"):
            raise ValueError(
                "Provide video_path or define info.video_path."
            )
    os.makedirs(info.processed_video_path, exist_ok=True)
    
    bodypart_data = h5_to_pd()
    raw_positions = {
        node_name: {
                        "x": bodypart_data[node_name]['x'],
                        "y": bodypart_data[node_name]['y']
                    }
            for node_name in node_names
    }
    processed_positions = {
        node_name: {
            "x": np.load(f"{info.data_path}/smoothRes/{node_name}.npz")['xnN'][0][0],
            "y": np.load(f"{info.data_path}/smoothRes/{node_name}.npz")['xnN'][3][0]
        }
        for node_name in node_names
    }
    
    # Generate one random color per node
    color_seed = 42
    rng = np.random.default_rng(color_seed)

    node_colors = {
        node_name: tuple(
            int(value)
            for value in rng.integers(50, 256, size=3)
        )
        for node_name in node_names
    }

    ## The tuple is BGR because OpenCV uses BGR rather than RGB.
    print("Node colors:")
    for node_name, color in node_colors.items():
        print(f"  {node_name}: {color}")

    # Open original video
    capture = cv2.VideoCapture(str(info.raw_video_path+'/'+info.video_name))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {info.raw_video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    tracking_frame_count = min(
        len(raw_df),
        len(processed_df),
    )

    if video_frame_count > 0:
        n_frames = min(video_frame_count, tracking_frame_count)
    else:
        n_frames = tracking_frame_count

    print(f"Video frames: {video_frame_count:,}")
    print(f"Raw position frames: {len(raw_df):,}")
    print(f"Processed position frames: {len(processed_df):,}")
    print(f"Frames to render: {n_frames:,}")
    
    # Initialize output video
    header_height = 64

    output_width = frame_width * 2
    output_height = frame_height + header_height

    writer = cv2.VideoWriter(
        str(info.processed_video_path+'/'+'compare_raw_processed.mp4'),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.video_frame_rate,
        (output_width, output_height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not initialize video writer for: {info.processed_video_path}"
        )

    # Render video
    rendered_frames = 0

    try:
        for frame_index in range(n_frames):
            success, video_frame = capture.read()

            if not success:
                print(f"Video ended at frame {frame_index:,}")
                break

            # Create the two panels from the same video frame.
            left_panel = np.full(
                (output_height, frame_width, 3),
                255,
                dtype=np.uint8,
            )

            right_panel = np.full(
                (output_height, frame_width, 3),
                255,
                dtype=np.uint8,
            )

            left_panel[
                header_height:header_height + frame_height
            ] = video_frame

            right_panel[
                header_height:header_height + frame_height
            ] = video_frame

            draw_title(left_panel, "raw")
            draw_title(right_panel, "processed")

            draw_positions(
                left_panel,
                raw_positions,
                frame_index,
            )

            draw_positions(
                right_panel,
                processed_positions,
                frame_index,
            )

            combined_frame = np.hstack(
                [left_panel, right_panel]
            )

            # Vertical separator between panels
            cv2.line(
                combined_frame,
                (frame_width, 0),
                (frame_width, output_height - 1),
                (0, 0, 0),
                thickness=2,
            )

            writer.write(combined_frame)
            rendered_frames += 1

            if rendered_frames % 5000 == 0:
                print(
                    f"Rendered {rendered_frames:,} / "
                    f"{n_frames:,} frames"
                )

    finally:
        capture.release()
        writer.release()

    print(f"Rendered {rendered_frames:,} frames")
    print(f"Saved video to:\n{info.processed_video_path}")