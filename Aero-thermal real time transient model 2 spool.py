import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# 1. COUPLING HIGH-FIDELITY CORE PERFORMANCE MAPS (HPC, HPT FROM MODULES)
# =============================================================================
from hpc_map import HPC_PyCycle
from hpt_map import HPT_PyCycle

# =============================================================================
# 2. INLINED HIGH-CAPACITY IPC PERFORMANCE MAP (OPTION 2 DIRECT VECTOR SCALING)
# =============================================================================
ipc_alpha_grid = np.array([1.0])  
ipc_Nc_grid    = np.array([0.50, 0.65, 0.75, 0.85, 0.95, 1.00, 1.10])
ipc_Rline_grid = np.array([1.00, 1.50, 2.00, 2.50, 3.00])

# Baseline generic low-pressure compressor map efficiency matrix
ipc_eff_data = np.array([[[0.72, 0.75, 0.77, 0.76, 0.72], 
                          [0.76, 0.79, 0.81, 0.79, 0.74], 
                          [0.78, 0.82, 0.84, 0.81, 0.75], 
                          [0.80, 0.83, 0.85, 0.82, 0.76],
                          [0.79, 0.82, 0.84, 0.81, 0.74], 
                          [0.77, 0.80, 0.82, 0.79, 0.72],
                          [0.73, 0.76, 0.78, 0.74, 0.68]]])

# BASELINE GENERAL COMPRESSOR FLOW CAPACITY RAW DATA MATRIX (lbm/s)
ipc_Wc_base = np.array([[[11.2, 10.7, 10.2,  8.9,  7.5], 
                         [14.9, 14.4, 13.9, 12.5, 10.4], 
                         [18.1, 17.6, 17.1, 15.8, 13.6], 
                         [21.6, 21.1, 20.6, 18.5, 16.4],
                         [25.6, 25.0, 24.5, 22.4, 20.3], 
                         [28.3, 27.8, 27.3, 25.2, 23.0],
                         [30.7, 30.2, 29.7, 27.6, 25.5]]])

# DIRECT ARRAY CAPACITY SCALING (1.35x Multiplier Scaled on the Raw Flow Grid Numbers)
# This mathematically scales the map throat physical bounds to remove compressor choke
ipc_Wc_scaled = ipc_Wc_base * 1.85

interp_IPC_Wc  = RegularGridInterpolator((ipc_alpha_grid, ipc_Nc_grid, ipc_Rline_grid), ipc_Wc_scaled, bounds_error=False, fill_value=None)
interp_IPC_eff = RegularGridInterpolator((ipc_alpha_grid, ipc_Nc_grid, ipc_Rline_grid), ipc_eff_data, bounds_error=False, fill_value=None)

def IPC_With_Map(T24, P24, P25, NIP):
    """Evaluates intermediate compressor states via the scaled inlined raw map arrays."""
    theta = T24 / 288.15
    delta = P24 / 101325.0
    
    # BLEED LOGIC: Artificial pressure buffer to prevent map collapse
    # This acts as a bleed valve that prevents the compressor from hitting a 1.0 PR
    P25_effective = max(P25, P24 * 1.35) 
    PR_actual = P25_effective / max(1000.0, P24)
    
    
    design_speed_rpm = 8200.0  
    
    Nc_current = (NIP / design_speed_rpm) / np.sqrt(theta)
    
    # Inline evaluation routines to track running R-line indices
    Nc_clamped = np.clip(Nc_current, ipc_Nc_grid.min(), ipc_Nc_grid.max())
    
    # Analytical inverse solution to track grid operating Rline positions
    r_line_resolved = 1.0 + (PR_actual - 1.0) * 0.4
    r_line_clamped = np.clip(r_line_resolved, ipc_Rline_grid.min(), ipc_Rline_grid.max())
    
    eval_coords = np.array([1.0, Nc_clamped, r_line_clamped])
    
    Wc_lbs = float(interp_IPC_Wc(eval_coords)[0])
    eta_ipc = float(interp_IPC_eff(eval_coords)[0])
    
    # Convert map corrected parameters back to physical operating mass flow (kg/s)
    W24_kg_per_sec = (Wc_lbs * delta / np.sqrt(theta)) * 0.453592
    
    gamma_air = 1.40
    T25_new = T24 + T24 * (1.0 / eta_ipc) * ((PR_actual) ** ((gamma_air - 1.0) / gamma_air) - 1.0)
    
    IPCPW = W24_kg_per_sec * 1005.0 * (T25_new - T24)
    return W24_kg_per_sec, T25_new, IPCPW

# =============================================================================
# 3. INLINED HIGH-EXTRACTION IPT PERFORMANCE MAP DATA
# =============================================================================
ipt_alpha_grid = np.array([1.0])  
ipt_Np_grid    = np.array([50., 60.0, 70.0, 80.0, 90.0, 100.])
ipt_PR_grid    = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

ipt_eff_data = np.array([[[0.85, 0.87, 0.89, 0.90, 0.89, 0.87, 0.84, 0.80],
                          [0.87, 0.89, 0.91, 0.92, 0.90, 0.88, 0.85, 0.82],
                          [0.88, 0.90, 0.92, 0.93, 0.91, 0.89, 0.86, 0.83],
                          [0.89, 0.91, 0.93, 0.94, 0.92, 0.90, 0.87, 0.84],
                          [0.88, 0.90, 0.92, 0.93, 0.91, 0.89, 0.86, 0.83],
                          [0.86, 0.88, 0.89, 0.90, 0.88, 0.86, 0.83, 0.80]]])

ipt_Wp_data = np.array([[[22.2, 22.5, 22.7, 22.8, 22.9, 22.9, 23.0, 23.0],
                         [21.8, 22.1, 22.3, 22.4, 22.5, 22.5, 22.6, 22.6],
                         [21.2, 21.5, 21.7, 21.8, 21.9, 21.9, 22.0, 22.0],
                         [20.5, 20.8, 21.0, 21.1, 21.2, 21.2, 21.3, 21.3],
                         [19.8, 20.1, 20.3, 20.4, 20.5, 20.5, 20.6, 20.6],
                         [19.0, 19.3, 19.5, 19.6, 19.7, 19.7, 19.8, 19.8]]])

interp_IPT_Wp  = RegularGridInterpolator((ipt_alpha_grid, ipt_Np_grid, ipt_PR_grid), ipt_Wp_data, bounds_error=False, fill_value=None)
interp_IPT_eff = RegularGridInterpolator((ipt_alpha_grid, ipt_Np_grid, ipt_PR_grid), ipt_eff_data, bounds_error=False, fill_value=None)

def IPT_With_Map(T42, P40, P42, P44, NIP):
    """Evaluates the custom intermediate work-extraction turbine map."""
    theta = T42 / 288.15
    delta = P42 / 101325.0
    design_speed_rpm = 6500.0  
    Np_current = (NIP / design_speed_rpm) / np.sqrt(theta) * 100.0
    PR_actual = P42 / max(1000.0, P44)
    
    Np_clamped = np.clip(Np_current, ipt_Np_grid.min(), ipt_Np_grid.max())
    PR_clamped = np.clip(PR_actual, ipt_PR_grid.min(), ipt_PR_grid.max())
    
    eval_coords = np.array([1.0, Np_clamped, PR_clamped])
    Wp_lbs = float(interp_IPT_Wp(eval_coords)[0])
    eta_ipt = float(interp_IPT_eff(eval_coords)[0])
    
    W44_kg_per_sec = (Wp_lbs * delta / np.sqrt(theta)) * 0.453592
    T44_new = T42 - T42 * eta_ipt * (1.0 - (1.0 / PR_clamped) ** (0.33 / 1.33))
    IPTPW = W44_kg_per_sec * 1150.0 * (T42 - T44_new)
    return W44_kg_per_sec, T44_new, IPTPW

def HPC_With_Map_Wrapper(T25, P25, P30, NH):
    W25_new, T30_new, eta_hpc, r_line = HPC_PyCycle(T25, P25, P30, NH)
    HPCPW = W25_new * 1005.0 * (T30_new - T25)
    return W25_new, T30_new, HPCPW

def HPT_With_Map_Wrapper(T40, P40, P42, NH):
    W42_new, T42_new, eta_hpt = HPT_PyCycle(T40, P40, P42, NH)
    HPTPW = W42_new * 1150.0 * (T40 - T42_new)
    return W42_new, T42_new, HPTPW

# =============================================================================
# 4. RUNWAY TAKEOFF SYSTEM SPACE INITIALIZATION
# =============================================================================
T24_ambient = 275.0     
P24_ambient = 101325.0  
P44_amb = 101325.0     

V_flight = 0.0          
Aircraft_Mass = 1500.0  
Aerodynamic_Drag_Coeff = 0.22 

# Physical fluid volumes 
IPCV = 5.0   
HPCV = 0.5   
IPTV = 0.8   
dt = 0.010   

P25 = 101325.0 * 1.40  
P30 = 101325.0 * 4.40  
P42 = 101325.0 * 1.85  

NH = 8500.0           
NIP = 9000.0          
NHXJ  = 0.5 * 5000.0 * 0.05**2  
NIPXJ = 0.5 * 12000.0 * 0.07**2  

# Logging structures
time_history = []
t25_history, t30_history, t44_history = [], [], []
p25_history, p30_history, p42_history = [], [], []
nh_history, nip_history = [], []
w24_history, w25_history, w42_history = [], [], []
thrust_history, v_flight_history = [], []

# =============================================================================
# 5. GENERATE PROFILE TARGET CONTEXT FILE
# =============================================================================
csv_filename = "transient_profile.csv"

# Create a smooth ramp from 4s to 20s
times = list(range(121))
wf_profile = [0.25]*4 + [0.25 + (0.13 * (t-4)/16) for t in range(4, 20)] + [0.38]*101
t40_profile = [1000]*4 + [1000 + (200 * (t-4)/16) for t in range(4, 20)] + [1200]*101

pd.DataFrame({"Time": times, "Wf": wf_profile, "T40": t40_profile}).to_csv(csv_filename, index=False)
profile_df = pd.read_csv(csv_filename)

# =============================================================================
# 6. EXECUTING INTEGRATOR LOOP
# =============================================================================
print("Executing 120s Run with direct Array Scaling in internal IPC map arrays...")
current_T25 = T24_ambient * 1.15

for index, row in profile_df.iterrows():
    csv_time = row['Time']
    target_Wf = row['Wf']
    target_T40 = row['T40']
    
    substeps = int(1.0 / dt)
    
    for _ in range(substeps):
        # Forward ram flow effects
        a_speed_of_sound = np.sqrt(1.4 * 287.0 * T24_ambient)
        Mach = V_flight / a_speed_of_sound
        T24_total = T24_ambient * (1.0 + 0.2 * Mach**2)
        P24_total = P24_ambient * (1.0 + 0.2 * Mach**2)**3.5 * 0.97
        
        P40 = P30 * 0.96  
        
        # Aerodynamic component step passes
        W24, calculated_T25, IPCPW = IPC_With_Map(T24_total, P24_total, P25, NIP)
        current_T25 = calculated_T25 
        W25, T30, HPCPW = HPC_With_Map_Wrapper(current_T25, P25, P30, NH)
        
        W40, T42, HPTPW = HPT_With_Map_Wrapper(target_T40, P40, P42, NH)
        W42, T44, IPTPW = IPT_With_Map(T42, P40, P42, P44_amb, NIP)
        
        # Pneumatic storage updates
        dW25dt = W24 - W25
        P25 += (1.4 * 287.0 * current_T25 * dW25dt / IPCV) * dt
        
        W_in_core = W25 + target_Wf
        dW30dt = W_in_core - W40
        P30 += (1.4 * 287.0 * T30 * dW30dt / HPCV) * dt
        
        dW42dt = W40 - W42
        P42 += (1.33 * 287.0 * T42 * dW42dt / IPTV) * dt
        
        # Calculate the 'Pressure Ratio Health' factor
        # If PR drops below 1.5, we throttle the core demand (simulating a bleed valve)
        pr_health = np.clip(P25 / (P24_total * 1.5), 0.5, 1.0)
        effective_W25 = W25 * pr_health 
        
        # Use the effective_W25 for the pressure integration
        dW25dt = W24 - effective_W25
        P25 += (1.4 * 287.0 * current_T25 * dW25dt / IPCV) * dt
        
        # Now do the same for the core balance:
        dW30dt = (effective_W25 + target_Wf) - W40
        P30 += (1.4 * 287.0 * T30 * dW30dt / HPCV) * dt
        P42 = max(P42, P44_amb * 1.01)
        
        # Shaft speeds rotor dynamics
        NH += ((3600 * (HPTPW - HPCPW)) / (NH * NHXJ * (2 * np.pi)**2)) * dt
        NH = np.clip(NH, 4000.0, 14000.0)
        
        NIP += ((3600 * (IPTPW - IPCPW)) / (NIP * NIPXJ * (2 * np.pi)**2)) * dt
        NIP = np.clip(NIP, 3000.0, 50000.0)
        
        # Propulsion thrust calculations
        K_noz = 1.15  
        
        PR_nozzle = (P42 / P44_amb) * (1.0 / K_noz) # Reduced effective backpressure
        V_jet = np.sqrt(2.0 * 1150.0 * T44 * (1.0 - (1.0 / PR_nozzle)**(0.33/1.33))) if PR_nozzle > 1.0 else 0.0
        
        W_nozzle = W42 + target_Wf
        Net_Thrust = (W_nozzle * V_jet) - (W24 * V_flight)
        
        # Airframe translation momentum integration
        Aero_Drag_Force = 0.5 * 1.225 * V_flight**2 * Aerodynamic_Drag_Coeff * 12.0 
        V_flight += ((Net_Thrust - Aero_Drag_Force) / Aircraft_Mass) * dt
        V_flight = max(0.0, V_flight)

    # Secondary logging step triggers
    time_history.append(csv_time)
    t25_history.append(current_T25)
    t30_history.append(T30)
    t44_history.append(T44)
    p25_history.append(P25 / 101325.0)
    p30_history.append(P30 / 101325.0)
    p42_history.append(P42 / 101325.0)
    nh_history.append(NH)
    nip_history.append(NIP)
    w24_history.append(W24)
    w25_history.append(W25)
    w42_history.append(W42)
    thrust_history.append(Net_Thrust)
    v_flight_history.append(V_flight)
    
    if csv_time % 15 == 0 or csv_time <= 5:
        print(f"Time: {int(csv_time):3d}s | Vel: {V_flight:5.1f} m/s | W24: {W24:5.2f} kg/s | T25: {current_T25:5.1f}K | P25: {P25/101325.0:.2f} atm | Thrust: {Net_Thrust:6.1f} N")

# =============================================================================
# 7. FOUR-CHART PROPULSION DASHBOARD VISUALIZATION
# =============================================================================
print("\nRendering 4-Chart Propulsion Dashboard...")
fig, axs = plt.subplots(2, 2, figsize=(15, 11))

# --- Chart 1: Aero-Thermodynamic Engine Pressures & Temperatures ---
ax1_temp = axs[0, 0]
ax1_temp.plot(time_history, t25_history, 'b-', label='T25 Interstage Temp (K)', linewidth=2)
ax1_temp.plot(time_history, t30_history, 'r-', label='T30 Core Exit Temp (K)')
ax1_temp.set_xlabel('Time (Seconds)')
ax1_temp.set_ylabel('Temperature (Kelvin)', color='b')
ax1_temp.tick_params(axis='y', labelcolor='b')
ax1_temp.grid(True, linestyle=':', alpha=0.5)

ax1_pres = ax1_temp.twinx()
ax1_pres.plot(time_history, p25_history, 'b--', label='P25 Interstage Press (atm)', alpha=0.7, linewidth=2)
ax1_pres.plot(time_history, p30_history, 'r--', label='P30 Core Press (atm)', alpha=0.7)
ax1_pres.set_ylabel('Pressure (Atmospheres)', color='r')
ax1_pres.tick_params(axis='y', labelcolor='r')
lines1, labels1 = ax1_temp.get_legend_handles_labels()
lines2, labels2 = ax1_pres.get_legend_handles_labels()
ax1_temp.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax1_temp.set_title('1. Station Aero-Thermal Temperatures & Pressures')

# --- Chart 2: Rotor Kinematics & Shaft Speeds ---
axs[0, 1].plot(time_history, nh_history, 'g-', label='NH - High Pressure Spool', linewidth=2)
axs[0, 1].plot(time_history, nip_history, 'darkorange', linestyle='-', label='NIP - Intermediate Spool', linewidth=2)
axs[0, 1].set_xlabel('Time (Seconds)')
axs[0, 1].set_ylabel('Rotational Speed (RPM)')
axs[0, 1].grid(True, linestyle=':', alpha=0.5)
axs[0, 1].legend(loc='lower right')
axs[0, 1].set_title('2. Mechanical Rotational Shaft Speeds')

# --- Chart 3: Aerodynamic Component Mass Flows ---
axs[1, 0].plot(time_history, w24_history, 'teal', label='W24 - Front Intake Flow (Scaled Arrays)', linewidth=2.5)
axs[1, 0].plot(time_history, w25_history, 'royalblue', linestyle='--', label='W25 - HPC Core Flow')
axs[1, 0].plot(time_history, w42_history, 'crimson', linestyle=':', label='W42 - HPT Exhaust Flow')
axs[1, 0].set_xlabel('Time (Seconds)')
axs[1, 0].set_ylabel('Mass Flow Rate (kg/s)')
axs[1, 0].grid(True, linestyle=':', alpha=0.5)
axs[1, 0].legend(loc='upper left')
axs[1, 0].set_title('3. Component Passing Mass Flow Continuity')

# --- Chart 4: Propulsive Net Thrust & Airframe Velocities ---
ax4_vel = axs[1, 1]
ax4_vel.plot(time_history, v_flight_history, 'purple', label='Forward Airspeed (m/s)', linewidth=2.5)
ax4_vel.set_xlabel('Time (Seconds)')
ax4_vel.set_ylabel('Aircraft Velocity (m/s)', color='purple')
ax4_vel.tick_params(axis='y', labelcolor='purple')
ax4_vel.grid(True, linestyle=':', alpha=0.5)

ax4_thrs = ax4_vel.twinx()
ax4_thrs.plot(time_history, thrust_history, 'crimson', linestyle='-.', label='Net Thrust (N)', linewidth=2)
ax4_thrs.set_ylabel('Net Thrust (Newtons)', color='crimson')
ax4_thrs.tick_params(axis='y', labelcolor='crimson')
lines7, labels7 = ax4_vel.get_legend_handles_labels()
lines8, labels8 = ax4_thrs.get_legend_handles_labels()
ax4_vel.legend(lines7 + lines8, labels7 + labels8, loc='center right', fontsize=9)
ax4_vel.set_title('4. External Airframe Flight Performance')

plt.suptitle('Unified Gas Turbine Takeoff Dashboard (120s Sweep - Map Arrays Scaled Directly)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()