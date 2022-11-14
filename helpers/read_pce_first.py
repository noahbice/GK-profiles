import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def get_measurement_data(data_file, num_points):
    data = pd.read_excel(data_file, header=12, usecols=[0, 1, 2])
    data = data.to_numpy()
    I = data[:, 2]
    t = data[:, 0]
    I[np.where(I < 0.005 * np.amax(I))] = 0
    I /= np.amax(I)

    where_throw = np.where(I > 0.005)[0]
    where_throw_left = np.max([where_throw[0] - 5, 0])
    where_throw_right = np.max([where_throw[-1] + 5, 0])
    I = I[where_throw_left:where_throw_right]
    t = t[where_throw_left:where_throw_right]
    length = I.shape[0]
    t_large_signal = t[int(length * 0.3):int(length * 0.7)]
    I_large_signal = I[int(length * 0.3):int(length * 0.7)]

    square_wave = np.zeros(t.shape)
    t_middle = int(t.shape[0] / 2)
    if ('Z' in data_file) and ('29' in data_file):
        square_wave[t_middle - 4: t_middle + 5] = 1.
        float_spacing = 19.65
    else:
        square_wave[t_middle-3: t_middle + 4] = 1.
        float_spacing = 12.38
    short_conv = np.convolve(I_large_signal, square_wave, 'same')
    short_conv /= np.amax(short_conv)
    A = (short_conv - np.roll(short_conv, 1)) / short_conv
    B = (np.roll(short_conv, -1) - short_conv) / short_conv

    A = A > 0.1
    B = B > 0.1
    C = A * B
    high_signal_peaks, _ = find_peaks(C)
    # print((high_signal_peaks - np.roll(high_signal_peaks, 1))[1:-2])
    # float_spacing = np.mean((high_signal_peaks - np.roll(high_signal_peaks, 1))[1:-2])

    full_conv = np.convolve(I, square_wave, 'same')
    full_conv /= np.amax(full_conv)
    A_full = (full_conv - np.roll(full_conv, 1)) / full_conv
    B_full = (np.roll(full_conv, -1) - full_conv) / full_conv
    A_full = A_full > 0.1
    B_full = B_full > 0.1
    C_full = A_full * B_full
    full_peaks, _ = find_peaks(C_full)
    first_peak = full_peaks[0]

    peak_idxs = []
    for i in range(num_points):
        peak_idxs.append(first_peak + i * float_spacing)
    peak_idxs = np.array(peak_idxs, dtype='int')

    if ('Z' in data_file) and ('29' in data_file):
        peak_idxs += 9
    else:
        peak_idxs += 3


    true_Is = np.zeros(peak_idxs.shape[0])
    peak_idxs = sorted(peak_idxs)
    for i in range(len(peak_idxs)):
        idx = int(peak_idxs[i])
        I_ROI = I[idx-1: idx+2]

        # I_ROI = I_ROI[int(0.4 * s):int(0.6 * s)]
        middle = int(I_ROI.shape[0] / 2)
        I_ROI = I_ROI[middle - 1: middle + 1]


        ## mean of middle 10 (1 second)
        true_Is[i] = np.mean(I_ROI)

        ## median of middle 10 (1 second)
        # s = I_ROI.shape[0]
        # true_Is[i] = np.median(np.round(I_ROI[int(0.3*s):int(0.7*s)], 2))

        ## max
        # true_Is[i] = np.amax(I_ROI)

        ## numerically integrate entire pulse
        # true_Is[i] = np.sum(I_ROI) / (np.prod(I_ROI.shape))

    ### time to positions, assuming dwell positions are accurate
    if num_points == 109:
        ## 109 pts
        x1 = list([40, 45, 50, 55, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78])
        x2 = [80 + 0.5 * i for i in range(80)]
        x3 = [120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 145, 150, 155, 160]
        x1.extend(x2)
        x1.extend(x3)
    elif num_points == 81:
        x1 = np.linspace(80, 120, 81)
    elif num_points == 29:
        x1 = np.linspace(93, 107, 29)
    elif num_points == 61:
        x1 = np.linspace(85, 115, 61)
    elif num_points == 41:
        x1 = np.linspace(90, 110, 41)
    else:
        tail_dist = ((num_points-1) / 2) * 0.5
        x1 = np.linspace(100 - tail_dist, 100 + tail_dist, true_Is.shape[0])

    dwell_positions = np.array(x1)
    print('at end')
    plt.clf()
    plt.scatter(dwell_positions, true_Is)
    plt.show()
    return dwell_positions, true_Is
