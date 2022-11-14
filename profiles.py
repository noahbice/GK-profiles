import matplotlib.pyplot as plt
import numpy as np
import os
from PyPDF2 import PdfMerger
from datetime import date
import pandas as pd
import textwrap as twp
from sklearn.svm import SVR
from helpers.gamma import pass_gamma
from helpers.read_pce_first import get_measurement_data
from helpers.read_lgp import read_LGP
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings("ignore")
plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')

################
### SETTINGS ###
measurement_directory = 'P:/3-Machines/Gamma Knife/5K_ICON_2022_AfterJuly15th/3FlimlessScan/July27_2022_Profiles/'
profiles_directory = './lgp/npy/'
dta = 1.0  # mm
dd = 2.0  # percent
post_processing_threshold = 5 # 0.50
z_shift = -0.2  # mm
fit = 'SVM'  # options: 'SVM' for SVM fit or 'linear' for linear interpolation
################

top_row_color = 'gray'
cell_color = 'lightgray'
pass_color = 'lightgreen'
fail_color = 'tomato'
dd /= 100
regularization_C = 1e4
r2_boostrapping_train_split_percent = 0.85

measurements = os.listdir(measurement_directory)
measurements = sorted(measurements)
if len(measurements) == 9:
    measurements = measurements[3:] + measurements[:3]
num_pointss = []
for measurement in measurements:
    num_pointss.append(int(measurement.split('pts')[0].split('mm')[1][1:]))

tw_length = 20
out_table = pd.DataFrame(data={'': [], twp.fill('Measured Penumbra (Left) [mm]', tw_length): [],
                               twp.fill('Modeled Penumbra (Left) [mm]', tw_length): [],
                               twp.fill('$\Delta$ (Left) [mm]', tw_length): [],
                               twp.fill('Measured Penumbra (Right) [mm]', tw_length): [],
                               twp.fill('Modeled Penumbra (Right) [mm]', tw_length): [],
                               twp.fill('$\Delta$ (Right) [mm]', tw_length): [],
                               twp.fill('Measured FWHM [mm]', tw_length): [],
                               twp.fill('Modeled FWHM [mm]', tw_length): [],
                               twp.fill('$\Delta$ FWHM [mm]', tw_length): [],
                               twp.fill('Radiation Center Offset [mm]', tw_length): []})
                               # twp.fill('Gamma Pass Rate [%]', tw_length): []})
out_colors = []
column_colors = [top_row_color] * 12

if len(os.listdir('./LGP/npy/doses/')) < 9:
    read_LGP()

fig, axs = plt.subplots(3, 3)
for k in range(len(measurements)):
    profiles_file = ''
    if '4mm' in measurements[k]:
        profiles_file += '4mm'
        i = 0
    elif '8mm' in measurements[k]:
        profiles_file += '8mm'
        i = 1
    elif '16mm' in measurements[k]:
        profiles_file += '16mm'
        i = 2
    else:
        print('Problem loading files. Check naming convention and run read_lgp.py.')
    if 'X' in measurements[k]:
        profiles_file += 'x'
        j = 0
    elif 'Y' in measurements[k]:
        profiles_file += 'y'
        j = 1
    elif 'Z' in measurements[k]:
        profiles_file += 'z'
        j = 2
    else:
        print('Problem loading files. Check naming convention and run read_lgp.py')
    profiles_file += '.npy'
    lgp_pos = np.load(profiles_directory + 'positions/' + profiles_file)
    lgp_dose = np.load(profiles_directory + 'doses/' + profiles_file)
    table_colors = [top_row_color] + [cell_color] * 2
    # try:
    dwell_positions, true_Is = get_measurement_data(measurement_directory + measurements[k], num_pointss[k])
        # plt.clf()
        # plt.plot(dwell_positions, true_Is)
        # plt.show()
    # except:
    #     # print(measurement_directory + measurements[k] + ' not found')
    #     if k < len(measurements):
    #         axs[i, j].set_title('{}'.format(measurements[k].split('.')[0]))
    #         lgp_pos = np.load(profiles_directory + 'positions/' + profiles_file)
    #         lgp_dose = np.load(profiles_directory + 'doses/' + profiles_file)
    #         axs[i, j].plot(lgp_pos, lgp_dose, color='green', label='LGP beam model')
    #         axs[i, j].legend(fontsize=8, loc='upper right')
    #         axs[i, j].grid()
    #         axs[i, j].set_xlim([80., 120.])
    #
    #         k += 1
    #     continue
    print_table = [measurements[k].split('.')[0].split('mm')[0] + 'mm ' + measurements[k].split('.')[0].split('mm')[1][0]]
    # load reference LGP data

    # find ~0.5 mm shifts for noisy measurements, optional
    if len(dwell_positions) in [10000]:  #  always False
        normed_Is = true_Is / np.amax(true_Is)
        times = np.linspace(0, 1, normed_Is.shape[0])
        interper = interp1d(times, normed_Is)
        fine_res_time = np.linspace(0, 1, 5000)
        fine_res_Is = interper(fine_res_time)
        idx_left_half_max = np.where(np.abs(fine_res_Is[:2500] - (np.amax(fine_res_Is) / 2)) == np.amin(np.abs(fine_res_Is[:2500] - (np.amax(fine_res_Is) / 2))))[0]
        idx_right_half_max = np.where(np.abs(fine_res_Is[2500:] - (np.amax(fine_res_Is) / 2)) == np.amin(np.abs(fine_res_Is[2500:] - (np.amax(fine_res_Is) / 2))))[0]
        center_time = ((fine_res_time[2500 + idx_right_half_max] - fine_res_time[idx_left_half_max]) / 2) + fine_res_time[idx_left_half_max]
        center_idx = np.where(np.abs(times - center_time) == np.amin(np.abs(times - center_time)))[0]
        dwell_positions = np.linspace(100 - 0.5 * (((dwell_positions.shape[0] - 1) / 2)), 100 + 0.5 * (((dwell_positions.shape[0] - 1) / 2)), dwell_positions.shape[0])
        nominal_center_idx = np.where(dwell_positions == 100.)[0]
        shift = (nominal_center_idx - center_idx)[0] * 0.5
        dwell_positions += shift

    if 'z' in measurements[k].lower():
        lgp_pos += z_shift
        dwell_positions += z_shift

    # fitting
    svr_rbf = SVR(kernel='rbf', C=regularization_C, gamma=0.1)
    y_rbf = svr_rbf.fit(dwell_positions.reshape(-1, 1), true_Is).predict(dwell_positions.reshape(-1, 1))

    refit_Is = true_Is.copy()
    true_Is /= np.amax(true_Is)
    y_rbf /= np.amax(y_rbf)

    # post-processing -- assume any relative difference > 50% is processing error. ~ 1-2% of points
    min_position = np.amin(dwell_positions)
    max_position = np.amax(dwell_positions)
    refit_dwells = dwell_positions.copy()
    svrd = svr_rbf.predict(dwell_positions.reshape(-1, 1))
    svrd = svrd.reshape(-1) / np.amax(svrd)
    diff = np.abs((svrd - true_Is) / svrd)
    idxs = np.where(diff > post_processing_threshold)[0]
    dwell_positions = np.delete(dwell_positions, idxs)
    true_Is = np.delete(true_Is, idxs)
    refit_dwells = np.delete(refit_dwells, idxs)
    refit_Is = np.delete(refit_Is, idxs)
    linear_interper = interp1d(refit_dwells.flatten(), refit_Is.flatten() / np.amax(refit_Is.flatten()),
                               bounds_error=False, fill_value='extrapolate')
    y_rbf = svr_rbf.fit(refit_dwells.reshape(-1, 1), refit_Is).predict(dwell_positions.reshape(-1, 1))
    y_rbf /= np.amax(y_rbf)

    # clip plot axes
    axs[i, j].set_xlim([80, 120])

    r2s = []
    for _ in range(10):
        train_idx = np.random.choice(np.arange(true_Is.shape[0]),
                                     size=int(r2_boostrapping_train_split_percent * true_Is.shape[0]))
        test_idx = []
        for l in range(true_Is.shape[0]):
            if l not in train_idx:
                test_idx.append(l)
        train_x = true_Is[train_idx]
        test_x = true_Is[test_idx]
        train_y = true_Is[train_idx]
        test_y = true_Is[test_idx]
        bootstrap_svr = SVR(kernel='rbf', C=regularization_C, gamma=0.1).fit(train_x.reshape(-1, 1), train_y)
        r2s.append(bootstrap_svr.score(test_x.reshape(-1, 1), test_y))
    r2 = np.mean(r2s)

    if np.abs(true_Is.shape[0]) > 60:
        positions = np.linspace(80, 120, 12000)
    else:
        positions = np.linspace(dwell_positions[0], dwell_positions[-1], 12000)

    if fit == 'SVM':
        currents = svr_rbf.predict(positions.reshape(-1, 1))
    elif fit =='linear':
        currents = linear_interper(positions.flatten())
    else:
        currents = 0
        print('Fit model not recognized. Please use linear or SVM.')

    idx = np.where(currents == np.amax(currents))[0][0]
    center_position = positions[idx]
    center_val = currents[idx]
    first_loc = np.where(np.abs(currents[:idx] - (center_val / 2)) == np.amin(np.abs(currents[:idx] - (center_val / 2))))[0]
    second_loc = np.where(np.abs(currents[idx:] - (center_val / 2)) == np.amin(np.abs(currents[idx:] - (center_val / 2))))[0]
    second_loc += idx
    FWHM = positions[second_loc] - positions[first_loc]
    radiation_center = positions[int((second_loc - first_loc) / 2) + first_loc]

    # 80-20 widths
    currents = currents.flatten()
    left_20 = np.where(np.abs(currents[:idx] - (center_val * 0.20)) == np.amin(np.abs(currents[:idx] - (center_val * 0.20))))[0][0]
    left_80 = np.where(np.abs(currents[:idx] - (center_val * 0.80)) == np.amin(np.abs(currents[:idx] - (center_val * 0.80))))[0][0]
    right_20 = np.where(np.abs(currents[idx:] - (center_val * 0.20)) == np.amin(np.abs(currents[idx:] - (center_val * 0.20))))[0][0]
    right_80 = np.where(np.abs(currents[idx:] - (center_val * 0.80)) == np.amin(np.abs(currents[idx:] - (center_val * 0.80))))[0][0]
    right_20 += idx
    right_80 += idx
    left_80_20 = np.abs(positions[left_80] - positions[left_20])
    right_80_20 = np.abs(positions[right_80] - positions[right_20])

    # 80-20 penumbra LGP
    lgpd = lgp_dose.flatten()
    lgp_max = np.amax(lgpd)
    where_lgp_max = np.where(lgpd == lgp_max)[0][0]
    lgp_left_50 = np.where(np.abs(lgpd[:where_lgp_max] - (lgp_max * 0.50)) == np.amin(np.abs(lgpd[:where_lgp_max] - (lgp_max * 0.50))))[0][0]
    lgp_left_20 = np.where(np.abs(lgpd[:where_lgp_max] - (lgp_max * 0.20)) == np.amin(np.abs(lgpd[:where_lgp_max] - (lgp_max * 0.20))))[0][0]
    lgp_left_80 = np.where(np.abs(lgpd[:where_lgp_max] - (lgp_max * 0.80)) == np.amin(np.abs(lgpd[:where_lgp_max] - (lgp_max * 0.80))))[0][0]
    lgp_right_50 = np.where(np.abs(lgpd[where_lgp_max:] - (lgp_max * 0.50)) == np.amin(np.abs(lgpd[where_lgp_max:] - (lgp_max * 0.50))))[0][0]
    lgp_right_20 = np.where(np.abs(lgpd[where_lgp_max:] - (lgp_max * 0.20)) == np.amin(np.abs(lgpd[where_lgp_max:] - (lgp_max * 0.20))))[0][0]
    lgp_right_80 = np.where(np.abs(lgpd[where_lgp_max:] - (lgp_max * 0.80)) == np.amin(np.abs(lgpd[where_lgp_max:] - (lgp_max * 0.80))))[0][0]
    lgp_right_20 += where_lgp_max
    lgp_right_80 += where_lgp_max
    lgp_right_50 += where_lgp_max
    lgp_FWHM = np.abs(lgp_pos[lgp_left_50] - lgp_pos[lgp_right_50])
    lgp_left_80_20 = np.abs(lgp_pos[lgp_left_80] - lgp_pos[lgp_left_20])
    lgp_right_80_20 = np.abs(lgp_pos[lgp_right_80] - lgp_pos[lgp_right_20])


    # make strings add data to dataframe
    FWHM_string = 'FWHM: ' + str(np.round(FWHM[0], 2)) + ' mm'
    center_string = 'Radiation center: ' + str(np.round(radiation_center[0], 2)) + ' mm'
    r2_string = '$R^2 = $ {}'.format(np.round(r2, 2))
    left_80_20_string = 'Left 80-20: {} mm'.format(np.round(left_80_20, 2))
    right_80_20_string = 'Right 80-20: {} mm'.format(np.round(right_80_20, 2))
    print_table.append(np.round(left_80_20, 2))
    print_table.append(np.round(lgp_left_80_20, 2))
    print_table.append(np.round(left_80_20 - lgp_left_80_20, 2))
    if np.abs(left_80_20 - lgp_left_80_20) < 0.5:
        table_colors.append(pass_color)
    else:
        table_colors.append(fail_color)
    print_table.append(np.round(right_80_20, 2))
    print_table.append(np.round(lgp_right_80_20, 2))
    table_colors.append(cell_color)
    table_colors.append(cell_color)
    if np.abs(right_80_20 - lgp_right_80_20) < 0.5:
        table_colors.append(pass_color)
    else:
        table_colors.append(fail_color)
    print_table.append(np.round(right_80_20 - lgp_right_80_20, 2))
    print_table.append(np.round(FWHM[0], 2))
    print_table.append(np.round(lgp_FWHM, 2))
    table_colors.append(cell_color)
    table_colors.append(cell_color)
    print_table.append(np.round(FWHM[0] - lgp_FWHM, 2))
    if np.abs(FWHM[0] - lgp_FWHM) < 1.2:
        table_colors.append(pass_color)
    else:
        table_colors.append(fail_color)
    print_table.append(np.round(radiation_center[0] - 100, 2))
    if np.abs(radiation_center[0] - 100) < 0.3:
        table_colors.append(pass_color)
    else:
        table_colors.append(fail_color)


    # get gamma pass rate
    gx = np.linspace(min_position, max_position, 2 * dwell_positions.shape[0])
    gd = svr_rbf.predict(gx.reshape(-1, 1))
    gd = gd.reshape(-1) / np.amax(gd)
    ds_lgp_dose = []
    for position in gx:
        idx = (np.abs(lgp_pos - position)).argmin()
        ds_lgp_dose.append(lgp_dose[idx])
    ds_lgp_dose = np.array(ds_lgp_dose)
    gamma = pass_gamma(gd, ds_lgp_dose, dta=(dta * 4.0), dd=dd)  # 1%/1mm
    gamma = 1 - ((gamma > 1.0).astype('int').sum() / np.prod(gamma.shape))
    gamma_string = '$\gamma$ {}%/{} mm: {}%'.format(int(dd * 100), int(dta), np.round(gamma * 100, 1))
    # print_table.append(np.round(gamma * 100, 1))
    # if np.abs(np.round(gamma * 100, 1)) > 90.0:
    #     table_colors.append(pass_color)
    # else:
    #     table_colors.append(fail_color)

    if fit == 'SVM':
        fit_label = 'SVM fit'
    elif fit == 'linear':
        fit_label = 'Linear interpolation'
    axs[i, j].plot(lgp_pos, lgp_dose, color='green', label='LGP beam model')
    axs[i, j].scatter(dwell_positions, true_Is, s=17, color='darkorange', label='PTW Microdiamond')
    axs[i, j].plot(dwell_positions, y_rbf, color='navy', label=fit_label)
    axs[i, j].grid()
    axs[i, j].set_title(measurements[k].split('mm')[0] + ' mm ' + measurements[k].split('mm')[1][0] + ' (' +
                        measurements[k].split('mm')[1][1:].split('pts')[0] + ' points)')
    axs[i, j].text(0.97, 0.67, FWHM_string, fontsize=8, transform=axs[i, j].transAxes, ha='right')
    axs[i, j].text(0.97, 0.72, center_string, fontsize=8, transform=axs[i, j].transAxes, ha='right')
    # axs[i, j].text(0.97, 0.47, r2_string, fontsize=8, transform=axs[i, j].transAxes, ha='right')
    axs[i, j].text(0.97, 0.62, left_80_20_string, fontsize=8, transform=axs[i, j].transAxes, ha='right')
    axs[i, j].text(0.97, 0.57, right_80_20_string, fontsize=8, transform=axs[i, j].transAxes, ha='right')
    # axs[i, j].text(0.97, 0.57, gamma_string, fontsize=8, transform=axs[i, j].transAxes, ha='right')
    axs[i, j].legend(fontsize=8, loc='upper right')
    k += 1
    out_table.loc[len(out_table)] = print_table
    out_colors.append(table_colors)

axs[2, 1].set_xlabel('Position [mm]', fontsize=14)
axs[1, 0].set_ylabel('Relative Dose', fontsize=14)
scale = 1.8
fig.set_size_inches(11 * scale, 8.5 * scale)
plt.savefig('./outputs/3x3-profiles.pdf', dpi='figure', format='pdf', pad_inches=0.1)
# plt.show()
plt.close()

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
# ax.axis('tight')
table = ax.table(cellText=out_table.values, colLabels=out_table.columns, loc='center', cellColours=out_colors, cellLoc='center', colColours=column_colors)
for (row, col), cell in table.get_celld().items():
  if (row == 0) or (col == -1):
    cell.set_text_props(fontproperties=FontProperties(weight='bold'))
  if col == 0:
      cell.set_text_props(fontproperties=FontProperties(weight='bold'))
# table.set_fontsize(12)
table.auto_set_font_size(False)
for cell in table._cells:
    if cell[0] == 0 or cell[1] == 0:
        table._cells[cell].set_fontsize(7)
    else:
        table._cells[cell].set_fontsize(10)

# table.auto_set_column_width(col=list(range(len(out_table.columns))))
fig.set_size_inches(11 * scale, 8.5 * scale)
ax.text(0, 0.225, 'TG-178 Toleraces:', weight='bold', fontsize=14)
ax.text(0, 0.2, 'FWHM = 1.2 mm, Center = 0.3 mm, Penumbra = 0.5 mm', fontsize=14)
plt.savefig('./outputs/table.pdf', dpi='figure', format='pdf', pad_inches=0.1)
# plt.show()

try:
    merge_files = ['./outputs/3x3-profiles.pdf', './outputs/table.pdf', './outputs/LGP-beam-model.pdf']
    datestr = date.today().strftime('%b-%d-%Y')
    merger = PdfMerger()
    for pdf in merge_files:
        merger.append(pdf)
    merger.write('GK-profiles_' + datestr + '.pdf')
except:
    print('Access not granted to PDF. Please close the file and try again.')





