import numpy as np
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# 1. EXPANDED PYCYCLE GRID DEFINITIONS
# =============================================================================
alpha_grid = np.array([0.0])  # Static IGV angle baseline
Nc_grid    = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10])
Rline_grid = np.array([1.00, 1.50, 2.00, 2.50, 3.00])

# Aerodynamic Corrected Flow Table (lbm/s)
Wc_data = np.array([[[15.2, 14.5, 13.8, 12.0, 10.1], 
                     [20.1, 19.4, 18.7, 16.9, 14.0], 
                     [24.5, 23.8, 23.1, 21.3, 18.4], 
                     [29.2, 28.5, 27.8, 25.0, 22.1],
                     [34.5, 33.8, 33.1, 30.3, 27.4], 
                     [38.2, 37.5, 36.8, 34.0, 31.1],
                     [41.5, 40.8, 40.1, 37.3, 34.4]]])

# Isentropic Efficiency Table
eff_data = np.array([[[0.72, 0.74, 0.75, 0.73, 0.68], 
                      [0.75, 0.77, 0.78, 0.76, 0.71],
                      [0.78, 0.81, 0.82, 0.79, 0.74], 
                      [0.81, 0.84, 0.85, 0.82, 0.77],
                      [0.82, 0.85, 0.86, 0.83, 0.78], 
                      [0.80, 0.83, 0.84, 0.81, 0.76],
                      [0.77, 0.79, 0.80, 0.78, 0.72]]])

# REWRITTEN PR ARRAY: High-capacity matrix to capture high-pressure core transients
# Rows = Speed Lines (0.50 to 1.10), Columns = R-Lines (1.00 to 3.00)
PR_data = np.array([[[1.50, 1.42, 1.35, 1.25, 1.15], 
                     [2.10, 1.95, 1.80, 1.65, 1.45],
                     [2.90, 2.70, 2.50, 2.25, 1.95], 
                     [3.80, 3.55, 3.30, 3.00, 2.60],
                     [4.80, 4.50, 4.15, 3.75, 3.25], 
                     [5.90, 5.50, 5.05, 4.55, 3.90],
                     [6.80, 6.35, 5.80, 5.20, 4.45]]])

# Build the Interpolators
interp_Wc  = RegularGridInterpolator((alpha_grid, Nc_grid, Rline_grid), Wc_data, bounds_error=False, fill_value=None)
interp_eff = RegularGridInterpolator((alpha_grid, Nc_grid, Rline_grid), eff_data, bounds_error=False, fill_value=None)
interp_PR  = RegularGridInterpolator((alpha_grid, Nc_grid, Rline_grid), PR_data, bounds_error=False, fill_value=None)

# =============================================================================
# 2. MATCHING EVALUATION FUNCTION
# =============================================================================
def IPC_PyCycle(T24, P24, P25, NIP):
    """
    Evaluates intermediate pressure compressor properties from the rewritten data grid.
    Includes explicit efficiency scaling protections.
    """
    theta = T24 / 288.15
    delta = P24 / 101325.0
    design_speed_rpm = 6500.0  
    
    Nc_current = (NIP / design_speed_rpm) / np.sqrt(theta)
    Nc_clamped = np.clip(Nc_current, Nc_grid.min(), Nc_grid.max())
    alpha_current = 0.0
    PR_actual = P25 / P24
    
    # Inline R-line Search Matching Logic
    pr_samples = [float(interp_PR([alpha_current, Nc_clamped, r])[0]) for r in Rline_grid]
    
    if PR_actual <= pr_samples[-1]:
        # Engine is running normal or slightly stalled: map cleanly inside array bounds
        Rline_resolved = np.interp(PR_actual, pr_samples[::-1], Rline_grid[::-1])
    else:
        # Extreme pressure load: clamp to the maximum performance limit line
        Rline_resolved = Rline_grid.min()

    eval_coords = np.array([alpha_current, Nc_clamped, Rline_resolved])
    Wc_lbs = float(interp_Wc(eval_coords)[0])
    eta_ipc = float(interp_eff(eval_coords)[0])
    
    # Safety Filter: Prevent out-of-bounds mathematical efficiency cancellation
    eta_ipc = np.clip(eta_ipc, 0.55, 0.88)
    
    # Mass flow units conversion scaling (lbm/s -> kg/s)
    W25_kg_per_sec = (Wc_lbs * delta / np.sqrt(theta)) * 0.453592
    
    # Thermodynamic calculation for true physical temperature rise
    gamma_air = 1.4
    T25_new = T24 * (1.0 + (1.0 / eta_ipc) * ((PR_actual)**((gamma_air - 1.0) / gamma_air) - 1.0))
    
    return W25_kg_per_sec, T25_new, eta_ipc, Rline_resolved