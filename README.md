# Gas Turbine Transient Performance Model

This repository contains a high-fidelity, real-time aero-thermal transient simulation of a two-spool gas turbine engine. The model uses interpolated performance maps for the IPC (Intermediate Pressure Compressor), HPC (High-Pressure Compressor), and IPT (Intermediate Pressure Turbine) to simulate engine spool-up and flight dynamics.

## Key Features
* **Transient Integration:** Uses a mass-flow balance integrator to simulate engine acceleration transients over 120 seconds.
* **Component Mapping:** Employs `scipy.interpolate.RegularGridInterpolator` for accurate performance map lookups (IPC/HPC/HPT).
* **Transient Stability:** Includes active transient bleed/bypass logic to prevent compressor stall/surge during rapid fuel ramps.
* **Flight Dynamics:** Calculates net thrust and aircraft velocity based on engine transient state.

## Prerequisites
To run this simulation, you will need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

## Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/gas-turbine-transient-model.git](https://github.com/yourusername/gas-turbine-transient-model.git)
