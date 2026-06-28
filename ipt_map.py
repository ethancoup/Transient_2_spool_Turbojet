import numpy as np
from scipy.interpolate import RegularGridInterpolator

alpha_grid = np.array([1.0])  
Np_grid    = np.array([50., 60.0, 70.0, 80.0, 90.0, 100.])
PR_grid    = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

# TWEAKED EFFICIENCY: Increased the efficiency floor and peaking limits (up to 92%)
# This forces the turbine to extract more physical work to spin up the IP spool
eff_data = np.array([[[0.85, 0.87, 0.89, 0.90, 0.89, 0.87, 0.84, 0.80],
                      [0.87, 0.89, 0.91, 0.92, 0.90, 0.88, 0.85, 0.82],
                      [0.88, 0.90, 0.92, 0.93, 0.91, 0.89, 0.86, 0.83],
                      [0.89, 0.91, 0.93, 0.94, 0.92, 0.90, 0.87, 0.84],
                      [0.88, 0.90, 0.92, 0.93, 0.91, 0.89, 0.86, 0.83],
                      [0.86, 0.88, 0.89, 0.90, 0.88, 0.86, 0.83, 0.80]]])

# TWEAKED MASS FLOW: Increased base capacity values to swallow the core gas expansion 
# smoothly, which keeps upstream pressures balanced
Wp_data = np.array([[[22.2, 22.5, 22.7, 22.8, 22.9, 22.9, 23.0, 23.0],
                     [21.8, 22.1, 22.3, 22.4, 22.5, 22.5, 22.6, 22.6],
                     [21.2, 21.5, 21.7, 21.8, 21.9, 21.9, 22.0, 22.0],
                     [20.5, 20.8, 21.0, 21.1, 21.2, 21.2, 21.3, 21.3],
                     [19.8, 20.1, 20.3, 20.4, 20.5, 20.5, 20.6, 20.6],
                     [19.0, 19.3, 19.5, 19.6, 19.7, 19.7, 19.8, 19.8]]])

interp_Wp  = RegularGridInterpolator((alpha_grid, Np_grid, PR_grid), Wp_data, bounds_error=False, fill_value=None)
interp_eff = RegularGridInterpolator((alpha_grid, Np_grid, PR_grid), eff_data, bounds_error=False, fill_value=None)

def IPT_PyCycle(T42, P40, P42, P44, NIP):
    theta = T42 / 288.15
    delta = P42 / 101325.0
    design_speed_rpm = 6500.0  
    
    Np_current = (NIP / design_speed_rpm) / np.sqrt(theta) * 100.0
    PR_actual = P42 / max(1000.0, P44)
    
    Np_clamped = np.clip(Np_current, Np_grid.min(), Np_grid.max())
    PR_clamped = np.clip(PR_actual, PR_grid.min(), PR_grid.max())
    alpha_current = 1.0 
    
    eval_coords = np.array([alpha_current, Np_clamped, PR_clamped])
    Wp_lbs = float(interp_Wp(eval_coords)[0])
    eta_ipt = float(interp_eff(eval_coords)[0])
    
    W44_kg_per_sec = (Wp_lbs * delta / np.sqrt(theta)) * 0.453592
    gamma_gas = 1.33  
    T44_new = T42 - T42 * eta_ipt * (1.0 - (1.0 / PR_clamped) ** ((gamma_gas - 1.0) / gamma_gas))
    
    return W44_kg_per_sec, T44_new, eta_ipt