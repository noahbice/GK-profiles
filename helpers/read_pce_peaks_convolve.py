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

    square_wave = np.zeros(t.shape)
    t_middle = int(t.shape[0] / 2)
    if ('Z' in data_file) and ('29' in data_file):
        square_wave[t_middle - 4: t_middle + 5] = 1.
    else:
        square_wave[t_middle-3: t_middle + 4] = 1.
    conv = np.convolve(I, square_wave, 'same')
    conv /= np.amax(conv)

    # plt.clf()
    # plt.plot(t, I, label='current')
    # plt.plot(t, square_wave, label='square wave')
    # plt.plot(t, conv, label='convolution')
    # plt.legend()
    # plt.xlabel('time')
    # plt.ylabel('current')
    # plt.show()

    A = (conv - np.roll(conv, 1)) / conv
    B = (np.roll(conv, -1) - conv) / conv

    plt.clf()
    plt.plot(t, A)
    plt.plot(t, B)
    plt.show()

    A = A > 0.1
    B = B > 0.1
    C = A * B
    peak_idxs, _ = find_peaks(C)

    idx_spacing = int(np.median((peak_idxs - np.roll(peak_idxs, 1))[1:-2]))
    float_spacing = np.mean((peak_idxs - np.roll(peak_idxs, 1))[1:-2])
    # idx_spacing = 12  # measurements 6 seconds apart
    if len(peak_idxs) == num_points:
        pass
    to_add = []

    for i in range(len(peak_idxs) - 1):
        spread = ((peak_idxs[i + 1] - peak_idxs[i]) / idx_spacing)
        if spread > 1.5:
            print(spread, peak_idxs[i])
            to_add.append(peak_idxs[i] + idx_spacing)
        if spread > 2.5:
            to_add.append(peak_idxs[i] + 2 * idx_spacing)
        if spread > 3.5:
            to_add.append(peak_idxs[i] + 3 * idx_spacing)
    print(to_add, t[to_add])
    # peak_idxs = np.append(peak_idxs, to_add)

    plt.clf()
    to_remove = []
    if len(peak_idxs) > num_points:
        for idx in range(len(peak_idxs) - 1):
            if (peak_idxs[idx + 1] - peak_idxs[idx]) < (idx_spacing / 2):
                to_remove.append(idx + 1)
    peak_idxs = np.delete(peak_idxs, to_remove)
    peak_idxs = [peak_idxs[0]]
    for i in range(num_points):
        peak_idxs.append(peak_idxs[0] + i * idx_spacing)
    peak_idxs = np.array(peak_idxs, dtype='int')

    if len(peak_idxs) == 80:
        add_right = int(peak_idxs[-1] + idx_spacing)
        peak_idxs = np.append(peak_idxs, add_right)
        plt.scatter(t[add_right], [0], color='red')
    peak_idxs = peak_idxs.astype('int')

    plt.title(len(peak_idxs))
    plt.plot(t, I)
    plt.scatter(t[peak_idxs], np.zeros(peak_idxs.shape[0]), color='orange')
    plt.show()

    ########### DEAL WITH THIS
    # if len(peak_idxs) > num_points:
    #     print(data_file)
    #     for j in range(num_points - len(peak_idxs)):
    #         if j % 2 == 0:
    #             peak_idxs = np.append(peak_idxs, [peak_idxs[-1] + idx_spacing])
    #         else:
    #             peak_idxs = np.append(peak_idxs, [peak_idxs[0] - idx_spacing])
    #         j += 1
    # if len(peak_idxs) < num_points:
    #     for j in range(len(peak_idxs) - num_points):
    #         np.delete(peak_idxs, -1)



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
    return dwell_positions, true_Is
