# 1. Setup cluster
## Setup files
1. In your terminal, ssh to the cluster
   ```
    ssh netid@clustername.ycrc.yale.edu
   ```
   
2. Create a `dlc_correction` folder under your project directory:
   ```
    mkdir dlc_correction
   ```
3. Clone this repository:
   ```
    git clone https://github.com/zimoli02/freely_moving_data_correction.git
   ```
4. Go to the `freely_moving_data_correction` folder, inside you should see two subfolders, `Function` and `Script`:
   ``` 
    cd freely_moving_data_correction
   ```
5. I recommend creating a new folder `Data` to store the files if you are transferring from your local computer, or you can ignore this if you're directly computing deeplabcut on the same cluster and has data stored elsewhere.
   ```
    mkdir Data
   ```
## Setup python environment
1. If you already have deeplabcut environment on this cluster, ignore this step and modify line 19 in [runCorrection.sh](Script/runCorrection.sh) and [runSave.sh](Script/runSave.sh) to match with the name of your deeplabcut environment.
2. If you do not have deeplabcut environment installed, first request for an interactive node by typing this in your terminal:
```
salloc --partition=devel --time=1:00:00 --mem=8G --cpus-per-task=1
```
Then type the following:
```
module reset 
module load miniconda 
conda env create -f dlc-env.yml 
```
And you should be all set.

## Prepare raw data
The `.h5` file for point coordinates and the original video should be uploaded to the cluster.
If you are using the terminal, type this:
```
rsync -avhP \
    path_and_name_of_your_file.h5\
    path_and_name_of_your_video.mp4\
    netid@transfer-clustername.ycrc.yale.edu:project/freely_moving_data_correction/Data/
```

# Modify code
## Information about the session and data storage
Go to the [Script](Script) folder:
```
cd Script
```
Open the [INFORMATION.py](Script/INFORMATION.py) file:
```
nano INFORMATION.py
```
Then modify the details based on the information of your own data. After edition, press `Ctrl+O` and `Enter` to save your modifications, then press `Ctrl+X` to exit.
## Specify the nodes you want to process
Similarly, open the [NODE.txt](Script/NODE.txt) file:
```
nano NODE.txt
```
Then add only the node names you want to process. Make sure these node names appear in your raw deeplabcut `.h5` files.

# Run code
## Smoothing the positions of individual nodes
To run the code for smoothing the raw positions of the nodes you named, make sure you are under the `Script` directory, then type this in terminal:
```
N=$(awk 'END {print NR}' NODE.txt) 
sbatch --array="1-${N}" runCorrection.sh
```
This should create several parallel running jobs to save you some time.
You could check the `logs` folder under `Script` for output of these jobs that show the progress.
## Save processed data to .h5 file and overlay on video
To run the code for saving the data and overlaying both the raw and processed positions on the original video, type this in terminal:
```
sbatch runSave.sh
```

