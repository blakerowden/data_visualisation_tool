"""
Thesis Project - Data Visualisation Tool
.py file 1/1 - Main File
REIT4841 - Research & Development Methods and Practice
2023
"""

__author__ = "B.Rowden"

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
import os
import csv
import logging
from dataclasses import dataclass, field
from global_ import *

# Input the file names and the data to plot
FILE_NAME1 = "CSV_Data\experiment_name.csv"
DATA_1A = "I_stator_q"  # Name of the data in the csv file
DATA_1A_LABEL = DATA_1A  # Label for the data on the plot
DATA_1A_FILTER = False  # Apply a butterworth filter to the data
DATA_1B = "I_stator_d"  # Name of the seccond data set in the same csv file
DATA_1B_LABEL = DATA_1B
DATA_1B_FILTER = False

FILE_NAME2 = "CSV_Data\experiment_name2.csv"
DATA_2A = "I_rotor_q"
DATA_2A_LABEL = DATA_2A
DATA_2A_FILTER = True
DATA_2B = "I_rotor_d"
DATA_2B_LABEL = DATA_2B
DATA_2B_FILTER = True

# Change this to change the domain name of the data in the csv file
DOMAIN = "t_s"
TOTAL_TIME = 0.5

# Change these to change the title and y axis label
TITLE = "Example Title"
Y_AXIS_LABEL = "Current (p.u)"

# Filter settings
CUTOFF = 80
ORDER = 2

# Other settings
HIDE_Y_AXIS = False

####################################################


@dataclass(order=True)
class ExperimentalData:
    total_time: float
    experiment_data: np.array
    t_step: float = field(init=False)
    Fs: float = field(init=False)  # Sampling Frequency
    T: float = field(init=False)  # Period


def extact_data(select) -> ExperimentalData:
    dirname = os.path.dirname(__file__)
    if select == 1:
        fullpath = os.path.join(dirname, FILE_NAME1)
    elif select == 2:
        fullpath = os.path.join(dirname, FILE_NAME2)

    # Read the file into a list of rows
    with open(
        fullpath, "r", encoding="utf-8-sig"
    ) as infile:  # utf-8-sig automatically removes BOM
        reader = csv.reader(infile)
        rows = list(reader)

    # Write the updated rows back into the file
    with open(fullpath, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

    rawdata = np.genfromtxt(fullpath, dtype=float, delimiter=",", names=True)
    experiment = ExperimentalData(
        TOTAL_TIME,
        rawdata,
    )

    experiment.t_step = (
        experiment.experiment_data[DOMAIN][1] - experiment.experiment_data[DOMAIN][0]
    )
    experiment.Fs = 1 / experiment.t_step
    offset = experiment.experiment_data[DOMAIN][0]

    # Remove the offset from the time data
    experiment.experiment_data[DOMAIN] = experiment.experiment_data[DOMAIN] - offset

    return experiment


def plot_experiment(experiment, ax, key, label=None) -> None:
    ax.plot(
        experiment.experiment_data[DOMAIN],
        experiment.experiment_data[key],
        label=label,
    )
    ax.set_xlim(0, experiment.total_time)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.grid(True)
    ax.legend()
    ax.set_title(TITLE)


def plot_all_data(sim_experiment, prac_experiment) -> None:
    """
    Plot all the data based on the selected experiments.
    """

    # Create the figure
    fig, axs = plt.subplots(1, 1)

    # Plot the simulation data
    if FILE_NAME1 != "":
        if DATA_1A != "":
            plot_experiment(sim_experiment, axs, DATA_1A, label=DATA_1A_LABEL)
        if DATA_1B != "":
            plot_experiment(sim_experiment, axs, DATA_1B, label=DATA_1B_LABEL)

    # Plot the practical data
    if FILE_NAME2 != "":
        if DATA_2A != "":
            plot_experiment(prac_experiment, axs, DATA_2A, label=DATA_2A_LABEL)
        if DATA_2B != "":
            plot_experiment(prac_experiment, axs, DATA_2B, label=DATA_2B_LABEL)

    # Show the plot
    if HIDE_Y_AXIS:
        axs.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    plt.show()


def apply_butterworth(experiment, key) -> ExperimentalData:
    nyquist = 0.5 * experiment.Fs
    normal_cutoff = CUTOFF / nyquist
    # Get the filter coefficients
    b, a = butter(ORDER, normal_cutoff, btype="low", analog=False)
    # Apply the filter
    experiment.experiment_data[key] = filtfilt(b, a, experiment.experiment_data[key])
    return experiment


def main() -> None:
    """
    Main function for the application.
    """
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Extract the data from the CSV files
    if FILE_NAME1 != "":
        experiment1 = extact_data(1)
    if FILE_NAME2 != "":
        experiment2 = extact_data(2)

    # Apply the filters to the data
    if DATA_1A != "" and DATA_1A_FILTER:
        experiment1 = apply_butterworth(experiment1, DATA_1A)
    if DATA_1B != "" and DATA_1B_FILTER:
        experiment1 = apply_butterworth(experiment1, DATA_1B)
    if DATA_2A != "" and DATA_2A_FILTER:
        experiment2 = apply_butterworth(experiment2, DATA_2A)
    if DATA_2B != "" and DATA_2B_FILTER:
        experiment2 = apply_butterworth(experiment2, DATA_2B)

    # Plot the data
    plot_all_data(experiment1, experiment2)


if __name__ == "__main__":
    """
    Only run the main function if this file is run directly.
    """
    main()
