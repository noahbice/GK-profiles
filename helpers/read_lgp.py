import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')

def read_profiles(file):
    
    dcm = pydicom.dcmread(file)
    dose = dcm.pixel_array

    IPP = dcm.ImagePositionPatient
    spacing = np.array([0.5, 0.5, 0.5])

    z = np.arange(IPP[0], dose.shape[0]*spacing[0] + IPP[0], spacing[0])
    x = np.arange(IPP[1], dose.shape[1]*spacing[1] + IPP[1], spacing[1])
    y = np.arange(IPP[2], dose.shape[2]*spacing[2] + IPP[2], spacing[2])

    interpolator = RegularGridInterpolator((z, x, y), dose)

    z_min, z_max = np.amin(z) + 1, np.amax(z) - 1
    x_min, x_max = np.amin(x) + 1, np.amax(x) - 1
    y_min, y_max = np.amin(y) + 1, np.amax(y) - 1

    # (100, 100, 100) determined by max point dose with 4 mm collimator for x and y, z determined with FWHM
    center = np.where(dose == np.amax(dose))
    center_x = x[center[1][0]]
    center_y = y[center[2][0]]

    z_positions = np.linspace(z_min, z_max, 12000)
    z_coords = [[z, center_x, center_y] for z in z_positions]
    z_dose = interpolator(z_coords)
    z_dose /= np.amax(z_dose)

    idx_max = np.where(z_dose == np.amax(z_dose))[0][0]
    idx_left = np.where(np.abs(z_dose[:idx_max] - (np.amax(z_dose) / 2)) == np.amin(
        np.abs(z_dose[:idx_max] - (np.amax(z_dose) / 2))))[0]
    idx_right = np.where(np.abs(z_dose[idx_max:] - (np.amax(z_dose) / 2)) == np.amin(
        np.abs(z_dose[idx_max:] - (np.amax(z_dose) / 2))))[0]
    center_z = ((z_positions[idx_max + idx_right] - z_positions[idx_left]) / 2) + z_positions[idx_left][0]
    center_z = center_z[0]

    z_positions -= center_z - 100

    x_positions = np.linspace(x_min, x_max, 12000)
    x_coords = [[center_z, x, center_y] for x in x_positions]
    x_dose = interpolator(x_coords)
    x_dose /= np.amax(x_dose)
    x_positions -= center_x - 100

    y_positions = np.linspace(y_min, y_max, 12000)
    y_coords = [[center_z, center_x, y] for y in y_positions]
    y_dose = interpolator(y_coords)
    y_dose /= np.amax(y_dose)
    y_positions -= center_y - 100

    return x_positions, x_dose, y_positions, y_dose, z_positions, z_dose
    
def read_LGP():
    dir = './LGP/dcm/'
    files = os.listdir(dir)
    fig, axs = plt.subplots(1, 3)
    for file in files:
        x, xd, y, yd, z, zd = read_profiles(dir + file)

        np.save('./lgp/npy/positions/{}x.npy'.format(file.split('.')[0]), x)
        np.save('./lgp/npy/positions/{}y.npy'.format(file.split('.')[0]), y)
        np.save('./lgp/npy/positions/{}z.npy'.format(file.split('.')[0]), z)
        np.save('./lgp/npy/doses/{}x.npy'.format(file.split('.')[0]), xd)
        np.save('./lgp/npy/doses/{}y.npy'.format(file.split('.')[0]), yd)
        np.save('./lgp/npy/doses/{}z.npy'.format(file.split('.')[0]), zd)

        axs[0].plot(x, xd, label=file.split('.')[0])
        axs[1].plot(y, yd, label=file.split('.')[0])
        axs[2].plot(z, zd, label=file.split('.')[0])

    axs[0].set_title('X Profiles', fontsize=14)
    axs[1].set_title('Y Profiles', fontsize=14)
    axs[2].set_title('Z Profiles', fontsize=14)
    axs[0].set_xlabel('Position [mm]', fontsize=14)
    axs[1].set_xlabel('Position [mm]', fontsize=14)
    axs[2].set_xlabel('Position [mm]', fontsize=14)
    axs[0].set_ylabel('Relative Dose', fontsize=14)
    axs[1].set_ylabel('Relative Dose', fontsize=14)
    axs[2].set_ylabel('Relative Dose', fontsize=14)
    axs[1].legend()
    axs[0].legend()
    axs[2].legend()
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    scale = 2
    fig.set_size_inches(8.5 * scale, 3 * scale)
    plt.savefig('./outputs/LGP-beam-model.pdf', dpi='figure', format='pdf', pad_inches=0.1)