import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def get_measurement_data(data_file, num_points):
    data = pd.read_excel(data_file, header=12, usecols=[0, 1, 2])
    data = data.to_numpy()
    I = data[:, 2]

    # plt.clf()
    # plt.plot(I / np.amax(I))
    # plt.show()

    # for every "zero", check if i-1 and i+1 are zeros -- if unique_zeros > num_points, reduce threshold
    threshold = np.amax(I) * 0.2
    tries = 0
    learning_rate = 0.1
    last_num = [0, 0]
    threshes = []
    nums_found = []
    while True:
        wheres1 = np.where(I > threshold)[0]
        wheres2 = np.where(np.roll(I, 1) > threshold)[0]
        wheres3 = np.where(np.roll(I, -1) > threshold)[0]
        true_peaks = []
        for w in wheres1:
            if (w in wheres2) and (w in wheres3):
                true_peaks.append(w)
                # count += 1
        # print(true_peaks)
        # remove redundant true peak indices
        unique_true_peaks = []
        peak_start = []
        count = 0
        for wi in range(len(true_peaks) - 1):
            if true_peaks[wi] != (true_peaks[wi + 1] - 1):
                unique_true_peaks.append(true_peaks[wi])
                peak_start.append(true_peaks[wi] - count)
                count = 0
            else:
                count += 1

        num_found = len(unique_true_peaks)
        correction = 0
        if num_found == num_points:
            break
        if (num_found < num_points):
            if last_num[-1] == num_found:
                correction = (num_found - last_num[0]) * learning_rate
            else:
                correction = (num_found - last_num[-1]) * learning_rate
            threshold -= correction
        else:
            print('Data is f--d.')
            quit()
        print(tries, num_found, last_num, correction)
        print(tries, threshold, num_points, num_found, learning_rate)
        tries += 1
        if (tries % 200) == 0 and learning_rate > 1e-4:
            learning_rate *= 0.2
        if last_num[-1] != num_found:
            last_num.append(num_found)
            last_num = last_num[1:]
        nums_found.append(num_found)
        threshes.append(threshold)
        print(num_found, last_num[0])
        if tries == 1000:
            plt.clf()
            plt.plot(np.arange(1000), threshes)
            plt.plot(np.arange(1000), nums_found)
            plt.show()
        # if tries > 10:
        #     quit()
        # if (tries > 2000) and ((len(unique_true_peaks) % 2) == 0):
        #     unique_true_peaks = unique_true_peaks[:-1]
        #     print('Even number of points found for ' + data_file + '. Radiation center measurmement susceptible to 0.5 mm error.')
        #     break
    print(len(unique_true_peaks))
    # unique_true_peaks.append(I.shape[0])
    # peak_start.append(I.shape[0]-1)

    # call the current for each dwell position its maximum during the recording time
    print(len(peak_start), len(unique_true_peaks))

    last_true_peak = unique_true_peaks[0]
    true_Is = np.zeros((len(unique_true_peaks)))
    for i in range(len(unique_true_peaks) - 1):
        I_ROI = I[last_true_peak:unique_true_peaks[i + 1] - (unique_true_peaks[i + 1] - peak_start[i + 1])]
        s = I_ROI.shape[0]
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

        last_true_zero = unique_true_zeros[i + 1]

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
    return dwell_positions, true_Is
