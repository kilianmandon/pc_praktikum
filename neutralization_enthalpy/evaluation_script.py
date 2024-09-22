import numpy as np

# Given data
# Titration values (in mL), first two values discarded
titration_values = np.array([42.42, 42.11])
V_titration_error = 0.05  # in mL
V_acid_titration = 10.0

# Concentration of NaOH used for titration in mol/L
c_NaOH = 0.5  # assumed error-free

# Weight differences for acid container (in g)
mass_acid_before = 86.295
mass_acid_after = 65.704
density_acid = 1.03  # in g/mL (assumed, no error)
mass_error = 0.0005  # in g

# Data for calibration
U_min, U_max = 225, 230  # Voltage (in V)
I_min, I_max = 0.364, 0.375  # Current (in A)
t_calibration = 35.0  # Time in seconds
t_calibration_error = 0.2  # in s

# Thermoelectric potentials (in mV)
delta_phi_neutralization = 5.818559463986587e-05  # in mV
delta_phi_neutralization_error = 1.4143292631007191e-06  # in mV

delta_phi_calibration = 7.008752093802355e-05  # in mV
delta_phi_calibration_error = 1.4149690172454828e-06  # in mV

# Dilution experiment difference
delta_phi_dilution = 0.001685 - 0.0016822  # Difference in mV
phi_error = 0.000001  # phi error in mV
delta_phi_dilution_error = np.sqrt(2) * phi_error
print(f"Delta phi dilution (mV): {delta_phi_dilution}")
print(f"Delta phi dilution error (mV): {delta_phi_dilution_error}")

# Calculating mean and standard deviation for titration volumes
V_base = np.mean(titration_values)  # Mean titration volume
V_base_error = np.sqrt(np.std(titration_values, ddof=1)**2 / 2 + V_titration_error**2)  # Total error

print("Titration volumes mean (mL):", V_base)
print("Titration volumes error (mL):", V_base_error)

# Molar amount of acid in neutralization (in mol)
n_acid_titration = c_NaOH * V_base / 1000  # Convert mL to L for mol calculation
n_acid_error = n_acid_titration * np.sqrt((V_base_error / V_base)**2)  # Relative error propagation
print("Molar amount of acid (mol):", n_acid_titration)
print("Molar amount error (mol):", n_acid_error)

# Calculating acid concentration from titration
c_acid = n_acid_titration / V_acid_titration * 1000
c_acid_err = np.sqrt((n_acid_error / n_acid_titration)**2 + (V_titration_error / V_acid_titration)**2) * c_acid
print("Acid concentration (mol/L):", c_acid)
print("Acid concentration error (mol/L):", c_acid_err)

# Volume of acid by weight difference (in mL)
mass_difference = mass_acid_before - mass_acid_after
volume_acid_weight = mass_difference / density_acid  # Volume calculation by mass
volume_acid_weight_error = np.sqrt(2 * mass_error**2) / density_acid  # Error propagation
print("Volume of acid by weight (mL):", volume_acid_weight)
print("Volume error (mL):", volume_acid_weight_error)

# Calculating the molar amount for neutralization
n_acid_neutralization = volume_acid_weight * c_acid / 1000  # Convert mL to L
n_acid_neutralization_err = np.sqrt((volume_acid_weight_error / volume_acid_weight)**2 + (c_acid_err / c_acid)**2) * n_acid_neutralization
print("Total moles of acid in neutralization (mol):", n_acid_neutralization)
print("Total moles error (mol):", n_acid_neutralization_err)

# Electrical work during calibration (in J)
U_mean = (U_max + U_min) / 2
I_mean = (I_max + I_min) / 2
U_error = (U_max - U_min) / 2  # Absolute difference from mean
I_error = (I_max - I_min) / 2  # Absolute difference from mean
print("Mean Voltage (V):", U_mean)
print("Voltage error (V):", U_error)
print("Mean Current (A):", I_mean)
print("Current error (A):", I_error)

W_calibration = U_mean * I_mean * t_calibration  # Work formula
W_calibration_error = W_calibration * np.sqrt(
    (U_error / U_mean)**2 +
    (I_error / I_mean)**2 +
    (t_calibration_error / t_calibration)**2
)
print("Calibration work (J):", W_calibration)
print("Calibration work error (J):", W_calibration_error)

# Enthalpy during neutralization
delta_H_neutralization = -W_calibration * delta_phi_neutralization / delta_phi_calibration
delta_H_neutralization_error = abs(delta_H_neutralization) * np.sqrt(
    (W_calibration_error / W_calibration)**2 +
    (delta_phi_neutralization_error / delta_phi_neutralization)**2 +
    (delta_phi_calibration_error / delta_phi_calibration)**2
)
print("Neutralization enthalpy (J):", delta_H_neutralization)
print("Neutralization enthalpy error (J):", delta_H_neutralization_error)

# Enthalpy during dilution
delta_H_dilution = -W_calibration * (delta_phi_dilution / delta_phi_calibration)
delta_H_dilution_error = abs(delta_H_dilution) * np.sqrt(
    (W_calibration_error / W_calibration)**2 +
    (delta_phi_dilution_error / delta_phi_dilution)**2 +
    (delta_phi_calibration_error / delta_phi_calibration)**2
)
print("Dilution enthalpy (J):", delta_H_dilution)
print("Dilution enthalpy error (J):", delta_H_dilution_error)

# Molar enthalpy values
molar_enthalpy_neutralization = delta_H_neutralization / n_acid_neutralization
molar_enthalpy_neutralization_error = molar_enthalpy_neutralization * np.sqrt(
    (delta_H_neutralization_error / delta_H_neutralization)**2 +
    (n_acid_neutralization_err / n_acid_neutralization)**2
)
print("Molar neutralization enthalpy (J/mol):", molar_enthalpy_neutralization)
print("Molar neutralization enthalpy error (J/mol):", molar_enthalpy_neutralization_error)

molar_enthalpy_dilution = delta_H_dilution / n_acid_neutralization
molar_enthalpy_dilution_error = molar_enthalpy_dilution * np.sqrt(
    (delta_H_dilution_error / delta_H_dilution)**2 +
    (n_acid_neutralization_err / n_acid_neutralization)**2
)
print("Molar dilution enthalpy (J/mol):", molar_enthalpy_dilution)
print("Molar dilution enthalpy error (J/mol):", molar_enthalpy_dilution_error)

molar_only_neutralization = molar_enthalpy_neutralization - molar_enthalpy_dilution
molar_only_neutralization_err = np.sqrt(molar_enthalpy_neutralization_error**2 + molar_enthalpy_dilution_error**2)

print(f"Difference molar neutralization and dilution (J/mol): {molar_only_neutralization}")
print(f"Difference molar neutralization and dilution error (J/mol): {molar_only_neutralization_err}")