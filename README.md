# Blink Scripts

An assortment of utility scripts to deal with blink data

## Description

| Script               |  Description                                             |
|-----------------------|----------------------------------------------------------|
| `blink_statistics.py` | Calculate the precision and recall from annotation files |
| `concat_samples.py`   | Concatenate aspect ratio files                           |
| `get_samples.py`      | Create EAR files from annotation files                   |
| `plot_ear.py`         | Plot the EAR graph                                       |
| `train_svm.py`        | Train the SVM from EAR files                             |


## Workflow

```shell
# Create samples (this will take a while to complete)
# Tip: You can chain multiple of these commands together
./get_samples.py ../data/eyeblink8/1/26122013_223310_cam.avi sample_data/blink_01.txt sample_data/non_blink_01.txt

# Concat the output files together
./concat_samples.py

# Train the SVM
./train_svm.py sample_data/blink_dist.txt sample_data/non_blink_dist.txt

# Run the blink detector 
./blink --detector=landmark ./data/eyeblink8/1/26122013_223310_cam.avi

# Calculate statistics
./blink_statistucs.py ./file.tag ./data/eyeblink8/1/26122013_223310_cam.tag 
```

## Installation

You need Python 3, with pip installed on your system.

To install any dependencies, run:

```shell
pip install -r requirements.txt
```
