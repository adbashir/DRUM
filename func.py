
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
DEFAULT_FIG_SIZE = (20, 7)
DEFAULT_DPI = 100
COLOR_STATE = 'blue'
COLOR_LCS_SCORE = 'tab:red'
COLOR_CHANGE_POINT = 'red'

LABEL_STATE = 'state'
LABEL_LCS_SCORE = 'DRUM Score'
TITLE_RAW_DATA = 'Raw State Data Over Time'
TITLE_SEGMENTED_DATA = 'State Data Segmented into Windows'
TITLE_STATE_PER_WINDOW = 'State per Window'
TITLE_LCS_SCORE = 'LCS Score for Window Size'
TITLE_LCS_AND_STATE = 'DRUM Score and State per Window'

WINDOW_SIZE = 50
SLIDE_SIZE = 20


def load_data(file_path):
    return pd.read_csv(file_path)

def plot_raw_data(data):
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    plt.plot(data.index, data[LABEL_STATE], label=LABEL_STATE, color=COLOR_STATE, alpha=0.7)
    plt.xlabel('Time Index')
    plt.ylabel(LABEL_STATE)
    plt.title(TITLE_RAW_DATA)
    plt.legend()
    plt.show()

def segment_data(data, window_size=WINDOW_SIZE, slide_size=SLIDE_SIZE):
    windows = []
    start_idx = 0
    while start_idx + window_size <= len(data):
        windows.append(data.iloc[start_idx:start_idx + window_size])
        start_idx += slide_size
    return windows


def plot_windows(windows):
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    for i, window in enumerate(windows):
        if not window.empty:
            state_label = window[LABEL_STATE].mode()[0]  # Most frequent state in window
            label = f"State: {state_label} (Window {i})"
            plt.plot(window.index, window[LABEL_STATE], label=label, alpha=0.5)
    plt.xlabel('Time Index')
    plt.ylabel(LABEL_STATE)
    plt.title(TITLE_SEGMENTED_DATA.replace("Windows", f"Windows of Size {WINDOW_SIZE}"))
    plt.legend()
    plt.show()


def calculate_lcs(windows):
    lcs_scores = []
    majority_states = []

    for i in tqdm(range(len(windows)), desc='Calculating LCS for windows'):
        window = windows[i]
        majority_state = window[LABEL_STATE].mode()[0]
        majority_states.append(majority_state)

        if i > 0:
            prev_window = windows[i-1]
            delta_m = np.abs(window.mean() - prev_window.mean())
            delta_s = np.abs(window.std() - prev_window.std())
            delta_frm = np.zeros(len(window.columns) - 2)
            for j, col in enumerate(window.columns[:-2]):
                running_mean = window[col].rolling(window=len(window), min_periods=1).mean()
                prev_running_mean = prev_window[col].rolling(window=len(prev_window), min_periods=1).mean()
                intersections_current = count_intersections(window[col].values, running_mean.values)
                intersections_previous = count_intersections(prev_window[col].values, prev_running_mean.values)
                delta_frm[j] = np.abs(intersections_current - intersections_previous)

            lcs = (1/3) * delta_m.sum() + (1/3) * delta_s.sum() + (1/3) * delta_frm.sum()
            lcs_scores.append(lcs)
        else:
            lcs_scores.append(None)  # First window has no previous window to compare with

    # Normalize LCS scores
    valid_lcs_scores = [x for x in lcs_scores if x is not None]
    min_lcs = min(valid_lcs_scores) if valid_lcs_scores else 0
    max_lcs = max(valid_lcs_scores) if valid_lcs_scores else 1
    normalized_lcs = [(x - min_lcs) / (max_lcs - min_lcs) if x is not None else None for x in lcs_scores]

    return pd.DataFrame({
        'Window Index': range(len(windows)),
        'LCS Score': normalized_lcs,
        'Majority State': majority_states
    })

def find_state_change_points(majority_states):
    change_points = []
    for i in range(1, len(majority_states)):
        if majority_states[i] != majority_states[i-1]:
            change_points.append(i)  # Store the index of the window where the change occurs
    return change_points

def count_intersections(data, running_mean):
    intersections = 0
    for i in range(1, len(data)):
        if (data[i-1] > running_mean[i-1] and data[i] <= running_mean[i]) or \
           (data[i-1] < running_mean[i-1] and data[i] >= running_mean[i]):
            intersections += 1
    return intersections

def plot_lcs(lcs_results, change_points):
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    plt.plot(lcs_results['Window Index'], lcs_results['LCS Score'], label=LABEL_LCS_SCORE, color=COLOR_LCS_SCORE)
    for cp in change_points:
        plt.axvline(x=cp, color=COLOR_CHANGE_POINT, linestyle='dashed', alpha=0.6)
    plt.xlabel('Window Index')
    plt.ylabel(LABEL_LCS_SCORE)
    plt.title(TITLE_LCS_SCORE.replace("Window Size", f"Window Size {WINDOW_SIZE} with State Changes"))
    plt.legend()
    plt.show()

def plot_window_states(windows):
    states = [window[LABEL_STATE].mode()[0] for window in windows]
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    plt.plot(range(len(states)), states, marker='o', linestyle='-', color=COLOR_STATE, alpha=0.7)
    plt.xlabel('Window Index')
    plt.ylabel(LABEL_STATE)
    plt.title(TITLE_STATE_PER_WINDOW.replace("Window", f"Window (Size {WINDOW_SIZE})"))
    plt.grid(True)
    plt.show()

def plot_lcs_and_states(lcs_results, windows):
    states = [window[LABEL_STATE].mode()[0] for window in windows]

    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)

    # Plot LCS scores on the left y-axis
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel(LABEL_LCS_SCORE, color=COLOR_LCS_SCORE)
    ax1.plot(lcs_results['Window Index'], lcs_results['LCS Score'], color=COLOR_LCS_SCORE, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=COLOR_LCS_SCORE)

    # Create a second y-axis for state values
    ax2 = ax1.twinx()
    ax2.set_ylabel(LABEL_STATE, color=COLOR_STATE)
    ax2.plot(range(len(states)), states, color=COLOR_STATE, marker='s', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=COLOR_STATE)

    plt.title(TITLE_LCS_AND_STATE.replace("Window", f"Window (Size {WINDOW_SIZE})"))
    #fig.tight_layout()
    plt.show()

def plot_variables(file_path, num_vars, colors):
    """
    Plots multiple variables from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    num_vars (int): Number of variables to plot.
    colors (list): List of colors for each plot.
    """
    plt.style.use('classic')  # Set the plotting style
    data = pd.read_csv(file_path)  # Load data
    
    # Check if the number of colors and number of variables match
    if len(colors) < num_vars:
        raise ValueError("Number of colors provided is less than the number of variables to plot.")

    fig, axs = plt.subplots(num_vars, 1, figsize=(20, 2*num_vars))  # Adjust subplot arrangement and size

    for i in range(num_vars):
        ax = axs[i] if num_vars > 1 else axs  # Handle the case for a single subplot
        var_name = f'var{i+1}'
        if var_name not in data.columns:
            raise ValueError(f"{var_name} not found in the data.")
        ax.plot(data[var_name], color=colors[i], label=var_name)
        ax.set_title(f'Variable {i+1}', fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Index', fontsize=12)
        ax.legend(loc='upper right')

    plt.tight_layout()  # Optimize layout
    plt.show()