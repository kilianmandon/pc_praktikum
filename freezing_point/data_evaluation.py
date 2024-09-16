import numpy as np
import pandas as pd
from scipy.constants import zero_Celsius
import matplotlib.pyplot as plt

# Constants and given data
rho_formula_slope = -0.00022897  # g/mL per degree Celsius
rho_formula_intercept = 1.00277688  # g/mL
theta = 1858  # Kryoscopic constant in g*K/mol
T_room = 21.4  # Room temperature in Celsius
T_room_error = 0.1
rho_slope_error = 2.9e-6  # error in slope
rho_intercept_error = 6.5e-5  # error in intercept
V_solvent = 100.0  # mL
salt_masses = np.array([1.00, 2.00, 3.00])  # in g
salt_mass_error = 0.01  # in g
freezing_points_pure = np.array([0.58, 0.59, 0.59])  # in Celsius
freezing_points_solutions = {
    1.00: np.array([0.09, 0.09, 0.10]),
    2.00: np.array([-0.40, -0.40, -0.40]),
    3.00: np.array([-0.82, -0.87, -0.83])
}
device_error = 0.01  # Device error for temperature in Celsius

# Calculation of solvent density
def calculate_density(T):
    return rho_formula_slope * T + rho_formula_intercept

rho_solvent = calculate_density(T_room)

# Error propagation for solvent density
rho_error = np.sqrt((T_room_error * rho_formula_slope)**2 + rho_slope_error**2 * T_room**2 + rho_intercept_error**2)

# Calculation of solvent mass with error
m_solvent = V_solvent * rho_solvent
m_solvent_error = V_solvent * rho_error

# Displaying intermediate results
print(f"Room Temperature: {T_room:.2f} ± {T_room_error:.2f} °C")
print(f"Density of Solvent: {rho_solvent:.6f} ± {rho_error:.6f} g/mL")
print(f"Solvent Mass: {m_solvent:.2f} ± {m_solvent_error:.2f} g\n")

# Calculation of average freezing point and error for pure solvent
freezing_point_pure_avg = np.mean(freezing_points_pure)
freezing_point_pure_std = np.std(freezing_points_pure, ddof=1)  # Standard deviation for error
freezing_point_pure_error = np.sqrt(freezing_point_pure_std**2 + device_error**2)

print(f"Average Freezing Point (Pure Solvent): {freezing_point_pure_avg:.2f} ± {freezing_point_pure_error:.2f} °C\n")

# Function to calculate the difference in freezing points and its error
def calculate_delta_T(freezing_point_solution):
    avg_solution = np.mean(freezing_point_solution)
    std_solution = np.std(freezing_point_solution, ddof=1)
    solution_error = np.sqrt(std_solution**2 + device_error**2)
    delta_T = freezing_point_pure_avg - avg_solution
    delta_T_error = np.sqrt(freezing_point_pure_error**2 + solution_error**2)
    return delta_T, delta_T_error, avg_solution, solution_error

# Calculation of the molar mass of the salt and its error
def calculate_molar_mass(m_salt, delta_T, delta_T_error):
    m_ratio = m_salt / m_solvent
    m_ratio_error = m_ratio * np.sqrt((salt_mass_error / m_salt)**2 + (m_solvent_error / m_solvent)**2)
    M_2 = (theta * m_ratio) / delta_T * 2
    M_2_error = M_2 * np.sqrt((m_ratio_error / m_ratio)**2 + (delta_T_error / delta_T)**2)
    return M_2, M_2_error, m_ratio, m_ratio_error

molar_masses = []
errors = []
print("Intermediate Results for Each Solution:\n")
for m_salt, fp_solution in freezing_points_solutions.items():
    delta_T, delta_T_error, avg_solution, solution_error = calculate_delta_T(fp_solution)
    M_2, M_2_error, m_ratio, m_ratio_error = calculate_molar_mass(m_salt, delta_T, delta_T_error)
    molar_masses.append(M_2)
    errors.append(M_2_error)
    
    print(f"Salt Mass: {m_salt:.2f} g")
    print(f"Average Freezing Point (Solution): {avg_solution:.2f} ± {solution_error:.2f} °C")
    print(f"Freezing Point Difference ΔT: {delta_T:.2f} ± {delta_T_error:.2f} °C")
    print(f"Mass Ratio (m/m_solvent): {m_ratio:.4f} ± {m_ratio_error:.4f}")
    print(f"Molar Mass of Salt: {M_2:.2f} ± {M_2_error:.2f} g/mol\n")

# Calculate mean molar mass and its standard deviation and error
mean_molar_mass = np.mean(molar_masses)
std_molar_mass = np.std(molar_masses, ddof=1)
mean_molar_mass_error = np.sqrt(np.sum(np.array(errors)**2)) / len(errors)

# Displaying final results
print(f"Mean Molar Mass: {mean_molar_mass:.2f} g/mol ± {mean_molar_mass_error:.2f} g/mol")
print(f"Standard Deviation of Molar Mass: {std_molar_mass:.2f} g/mol\n")

# Plotting molar masses with errors over used solute mass
plt.errorbar(salt_masses, molar_masses, yerr=errors, fmt='o', capsize=5)
plt.xlabel('Salt Mass (g)')
plt.ylabel('Molar Mass (g/mol)')
plt.title('Molar Masses vs Salt Mass')
plt.grid(True)
plt.show()

# Output results
results = pd.DataFrame({
    'Salt Mass (g)': salt_masses,
    'Molar Mass (g/mol)': molar_masses,
    'Error (g/mol)': errors
})
results.to_csv('molar_masses_results.csv', index=False)