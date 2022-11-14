import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_measurement_data(data_file, num_points):
    data = pd.read_excel(data_file, header=12, usecols=[0, 1, 2])
    data = data.to_numpy()
    I = data[:, 2]


    # for every "zero", check if i-1 and i+1 are zeros -- if unique_zeros > num_points, reduce threshold
    threshold = 0.02
    tries = 0
    continuing = True
    multiply = True
    while True:
        wheres1 = np.where(I < threshold)[0]
        wheres2 = np.where(np.roll(I, 1) < threshold)[0]
        wheres3 = np.where(np.roll(I, -1) < threshold)[0]
        true_zeros = []
        for w in wheres1:
            if (w in wheres2) and (w in wheres3):
                true_zeros.append(w)
                # count += 1

        # remove redundant true zero indices
        unique_true_zeros = []
        zero_start = []
        count = 0
        for wi in range(len(true_zeros) - 1):
            if true_zeros[wi] != (true_zeros[wi + 1] - 1):
                unique_true_zeros.append(true_zeros[wi])
                zero_start.append(true_zeros[wi] - count)
                count = 0
            else:
                count += 1
        if len(unique_true_zeros) == num_points:
            break
        elif len(unique_true_zeros) < num_points:
            threshold += 0.03
        elif len(unique_true_zeros) > num_points:
            threshold -= 0.005
        # print(tries, threshold, num_points, len(unique_true_zeros))
        tries += 1
        if (continuing == False) and (len(unique_true_zeros) % 2) == 1:
            num_points = len(unique_true_zeros)
            break
        if (tries > 500) and ((len(unique_true_zeros) % 2) == 1):
            if multiply:
                threshold *= 0.2
                multiply = False
            else:
                threshold *= 1.03
            continuing = False
            num_points = len(unique_true_zeros)
            continue
        if (tries > 2000) and ((len(unique_true_zeros) % 2) == 0):
            unique_true_zeros = unique_true_zeros[:-1]
            print('Even number of points found for ' + data_file + '. Radiation center measurmement susceptible to 0.5 mm error.')
            break
    unique_true_zeros.append(I.shape[0])
    zero_start.append(I.shape[0]-1)

    # call the current for each dwell position its maximum during the recording time
    last_true_zero = unique_true_zeros[0]
    true_Is = np.zeros((len(unique_true_zeros) - 1))
    for i in range(len(unique_true_zeros) - 1):
        I_ROI = I[last_true_zero:unique_true_zeros[i + 1] - (unique_true_zeros[i + 1] - zero_start[i + 1])]
        middle = int(I_ROI.shape[0] / 2)
        I_ROI = I_ROI[middle - 1: middle + 2]

        ## mean of middle 3 (0.3 second)
        true_Is[i] = np.mean(I_ROI)
        # if i not in [len(unique_true_zeros), len(unique_true_zeros) - 1]:
        #     if (true_Is[i] - true_Is[i + 1]) / true_Is[i] > 0.5:
        #         print(I_ROI)

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
