import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d
import scipy.integrate as integrate

t = np.arange(0, 26)
# Values extracted from the image
phi_values = [
    -0.001742,
    -0.001742, -0.001743, -0.001743, -0.001744, -0.001745,
    -0.001803, -0.001803, -0.001802, -0.001802, -0.001802,
    -0.001810, -0.001834, -0.001850, -0.001860, -0.001865,
    -0.001868, -0.001870, -0.001871, -0.001872, -0.001872,
    -0.001872, -0.001872, -0.001872, -0.001872, -0.001872
]
phi_error = 0.000001

phi_values = -np.array(phi_values)

early_period = range(6)
mid_period = range(6, 11)
late_period = range(18, 26)

# Perform linear regression for each period
early_slope, early_intercept, _, _, _ = linregress(t[early_period], phi_values[early_period])
mid_slope, mid_intercept, _, _, _ = linregress(t[mid_period], phi_values[mid_period])
late_slope, late_intercept, _, _, _ = linregress(t[late_period], phi_values[late_period])

# Print linear regression values
print(f"Early Period Linear Regression: slope = {early_slope}, intercept = {early_intercept}")
print(f"Mid Period Linear Regression: slope = {mid_slope}, intercept = {mid_intercept}")
print(f"Late Period Linear Regression: slope = {late_slope}, intercept = {late_intercept}")

# Functions to calculate the linear fit values
def early_fit(x):
    return early_slope * x + early_intercept

def mid_fit(x):
    return mid_slope * x + mid_intercept

def late_fit(x):
    return late_slope * x + late_intercept

# Interpolation function for phi_values
interpolated_function = interp1d(t, phi_values, kind='linear', fill_value="extrapolate")

# Find the point where the areas are equal using interpolation and linspace
def find_transition_point(start, end, first_fit, second_fit, num_points=200):
    min_difference = float('inf')
    transition_point = start
    t_fine = np.linspace(start, end, num_points)
    
    for i in t_fine:
        # Calculate the area difference for the transition point
        early_area, _ = integrate.quad(lambda x: (first_fit(x) - interpolated_function(x)), start, i)
        # Measured values minus mid fit
        mid_area, _ = integrate.quad(lambda x: (interpolated_function(x) - second_fit(x)), i, end)
        early_area = early_area
        mid_area = -mid_area
        # print(f"{i}: {early_area:.3e} | {mid_area:.3e}")
        
        # Difference between areas
        difference = abs(early_area + mid_area)  # Should be zero when areas are equal
        if difference < min_difference:
            min_difference = difference
            transition_point = i
    
    return transition_point, second_fit(transition_point) - first_fit(transition_point)

# Determine the transition point with finer resolution
transition_point_neutralization, v_neut = find_transition_point(early_period[-1], mid_period[0], early_fit, mid_fit)
left_point_neutralization, v_neut_l = find_transition_point(early_period[-1], mid_period[0], lambda x: early_fit(x)+phi_error, lambda x: mid_fit(x)+phi_error)
right_point_neutralization, v_neut_r = find_transition_point(early_period[-1], mid_period[0], lambda x: early_fit(x)-phi_error, lambda x: mid_fit(x)-phi_error)
transition_point_calibration, v_cal = find_transition_point(mid_period[-1], late_period[0], mid_fit, late_fit)
left_point_calibration, v_cal_l = find_transition_point(mid_period[-1], late_period[0], lambda x: mid_fit(x)+phi_error, lambda x: late_fit(x)+phi_error)
right_point_calibration, v_cal_r = find_transition_point(mid_period[-1], late_period[0], lambda x: mid_fit(x)-phi_error, lambda x: late_fit(x)-phi_error)

v_neut_pos_error = max(abs(v_neut_l-v_neut), abs(v_neut_r-v_neut))
v_cal_pos_error = max(abs(v_cal_l-v_cal), abs(v_cal_r-v_cal))
v_neut_phi_err = np.sqrt(2) * phi_error
v_cal_phi_err = np.sqrt(2) * phi_error
v_neut_err = np.sqrt(v_neut_pos_error**2 + v_neut_phi_err**2)
v_cal_err = np.sqrt(v_cal_pos_error**2 + v_cal_phi_err**2)

print(f"Neutralization:")
print(f"Transition point: {transition_point_neutralization}")
print(v_neut)
print(f"Total error: {v_neut_err}")
print(f"By pos error: {v_neut_pos_error}")
# print(f"Left Transition point: {left_point_neutralization}")
# print(v_neut_l)
# print(f"Right Transition point: {right_point_neutralization}")
# print(v_neut_r)


print(f"Calibration:")
print(f"Transition point: {transition_point_calibration}")
print(v_cal)
print(f"Total error: {v_cal_err}")
print(f"By pos error: {v_cal_pos_error}")
# print(f"Left Transition point: {left_point_calibration}")
# print(v_cal_l)
# print(f"Right Transition point: {right_point_calibration}")
# print(v_cal_r)



# Plot the original data and linear fits
plt.plot(t, phi_values, label='Original Data')
plt.plot(np.linspace(0, 25, 50), early_fit(np.linspace(0, 25, 50)), 'r--', label='Early Period Fit')
plt.plot(np.linspace(0, 25, 50), mid_fit(np.linspace(0, 25, 50)), 'g--', label='Mid Period Fit')
plt.plot(np.linspace(0, 25, 50), late_fit(np.linspace(0, 25, 50)), 'g--', label='Late Period Fit')

# Highlight the transition point
plt.axvline(x=transition_point_neutralization, color='purple', linestyle='--', label='Transition Point Neutralization')
plt.axvline(x=transition_point_calibration, color='purple', linestyle='--', label='Transition Point Calibration')

plt.xlabel('Time (min)')
plt.ylabel('Phi (mV)')
plt.legend()
plt.show()

# Print the transition point
# print(f"The transition point is at t = {transition_point:.3f}")