# HemoDyn-AI: Research Reproducibility Protocol

This document provides a comprehensive, transparent reproducibility protocol for all computational fluid dynamics (CFD) simulations, physics-guided machine learning (PGML) models, regional error decompositions, and benchmark metrics presented in the HemoDyn-AI research project.

---

## 1. Research Overview & Scientific Objectives

Arterial stenosis (pathological narrowing of blood vessels) causes flow acceleration, elevated wall shear stress (WSS), pressure loss, and post-stenotic flow separation with vortex recirculation. While high-fidelity 3D computational fluid dynamics (CFD) based on the Navier-Stokes equations provides accurate hemodynamic fields, its computational cost limits real-time clinical workflows.

**HemoDyn-AI** addresses this challenge by establishing:
1. **A validated 4-case CFD benchmark** spanning healthy ($0\%$), mild ($30\%$), moderate ($50\%$), and severe ($70\%$) diameter stenosis.
2. **A physics-guided residual machine learning surrogate** that models the discrepancy between analytical Poiseuille/continuity theory and high-fidelity 3D Navier-Stokes solutions.
3. **Rigorous generalization benchmarks**, including spatial holdout validation across the narrowing, cross-geometry holdout validation on unseen stenosis severities, and reverse-flow diagnostics in post-stenotic recirculation zones.

---

## 2. Hemodynamic & Physical Parameters

| Parameter | Symbol | Value | Units | Physical Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Arterial Length** | $L$ | $120.0$ | $\text{mm}$ | Sufficient entry/exit length for developed flow |
| **Healthy Lumen Diameter** | $D_0$ | $4.0$ | $\text{mm}$ | Typical human coronary / femoral branch scale |
| **Stenosis Centerline** | $z_0$ | $60.0$ | $\text{mm}$ | Centered along vessel length |
| **Stenosis Gaussian Width** | $\sigma$ | $5.0$ | $\text{mm}$ | Smooth anatomical transition |
| **Blood Dynamic Viscosity** | $\mu$ | $0.004$ | $\text{Pa}\cdot\text{s}$ | Standard Newtonian blood viscosity |
| **Blood Density** | $\rho$ | $1060.0$ | $\text{kg/m}^3$ | Standard human whole blood density |
| **Steady Inflow Rate** | $Q$ | $1244.0$ | $\text{mm}^3/\text{s}$ | Mean physiologic flow rate ($1.244\text{ mL/s}$) |
| **Inlet Reynolds Number** | $\text{Re}_0$ | $\approx 99.4$ | dimensionless | Laminar steady baseline regime |

### Stenosis Profile Formulation

The local arterial radius $R(z)$ is parameterized as a function of axial distance $z$:

$$R(z) = R_0 \left[1 - S \cdot \exp\left(-\frac{1}{2}\left(\frac{z - z_0}{\sigma}\right)^2\right)\right]$$

where:
- $R_0 = 2.0\text{ mm}$ (baseline unconstricted radius)
- $S \in \{0.0, 0.30, 0.50, 0.70\}$ (nominal diameter reduction severity)
- $z_0 = 60.0\text{ mm}$
- $\sigma = 5.0\text{ mm}$

---

## 3. High-Fidelity CFD Reference Results

Simulations were computed using SimVascular svSolver (stabilized finite element formulation for incompressible Navier-Stokes). Spatial mesh convergence on the healthy vessel verified asymptotic convergence toward theoretical Poiseuille pressure gradient ($0.697\text{ Pa/mm}$ at ultrafine mesh).

### Four-Case Hemodynamic Benchmark Summary

| Geometry | Stenosis Severity | Peak Velocity ($\text{mm/s}$) | Pressure Drop $\Delta P$ ($\text{Pa}$) | Throat Mean WSS ($\text{Pa}$) | Max Peak WSS ($\text{Pa}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | $0\%$ | $204.04$ | $86.22$ | $0.64$ | $0.68$ |
| **Mild** | $30\%$ | $374.62$ | $109.95$ | $1.59$ | $2.25$ |
| **Moderate** | $50\%$ | $786.77$ | $211.94$ | $3.12$ | $5.73$ |
| **Severe** | $70\%$ | $2118.99$ | $1176.12$ | $9.56$ | $27.68$ |

*All metrics are verified from `results/four_case_comparison/hemodyn_four_case_summary.csv`.*

---

## 4. Machine Learning & Residual Surrogate Formulation

### Feature Space ($\mathbf{X} \in \mathbb{R}^7$)
1. $x$ — transverse Cartesian coordinate ($\text{mm}$)
2. $y$ — coronal Cartesian coordinate ($\text{mm}$)
3. $z$ — axial Cartesian coordinate along artery ($\text{mm}$)
4. $S$ — stenosis severity fraction ($0.0, 0.3, 0.5$)
5. $R(z)$ — local lumen radius ($\text{mm}$)
6. $r / R(z)$ — normalized radial position ($\in [0, 1]$)
7. $u_{\text{physics}}(r, z)$ — analytical Poiseuille/continuity velocity baseline ($\text{mm/s}$)

### Analytical Physics Baseline
$$u_{\text{physics}}(r, z) = \frac{2Q}{\pi R(z)^2} \max\left(1 - \frac{r^2}{R(z)^2}, 0\right)$$

### Surrogate Architectures Evaluated
1. **Analytical Physics Baseline**: Evaluates $u_{\text{physics}}$ directly without learning.
2. **Geometry-Only ML**: Standard histogram gradient boosting regressor fitted directly to $y = u_z$ using only geometric features $\mathbf{X}_{0:6}$.
3. **Physics-Guided ML (Residual Learning)**: The ML model is trained strictly on the physical discrepancy residual:
   $$\Delta u = u_{\text{CFD}} - u_{\text{physics}}$$
   The final prediction combines analytical theory with learned residual physics:
   $$\hat{u}_{\text{PGML}} = u_{\text{physics}} + \hat{f}_{\text{residual}}(\mathbf{X})$$

### Hyperparameters (Reproducible & Deterministic)
- **Algorithm**: `sklearn.ensemble.HistGradientBoostingRegressor`
- `max_iter`: 160
- `max_leaf_nodes`: 31
- `learning_rate`: 0.07
- `min_samples_leaf`: 40
- `l2_regularization`: 1.0
- `early_stopping`: False
- `random_state`: 42

---

## 5. Key Research Benchmark Results

### 1. Cross-Geometry Generalization (Leave-One-Geometry-Out)

| Held-out Geometry | Model | RMSE ($\text{mm/s}$) | MAE ($\text{mm/s}$) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Healthy ($0\%$)** | Physics baseline | $2.01$ | $1.42$ | $0.9991$ |
| | Geometry-only ML | $1.25$ | $0.82$ | $0.9996$ |
| | **Physics-guided ML** | **$1.12$** | **$0.73$** | **$0.9997$** |
| **Mild ($30\%$)** | Physics baseline | $8.33$ | $3.57$ | $0.9873$ |
| | Geometry-only ML | $20.98$ | $6.39$ | $0.9191$ |
| | **Physics-guided ML** | **$16.51$** | **$5.53$** | **$0.9499$** |
| **Moderate ($50\%$)** | Physics baseline | $23.30$ | $8.43$ | $0.9312$ |
| | Geometry-only ML | $29.21$ | $7.80$ | $0.8919$ |
| | **Physics-guided ML** | **$17.29$** | **$5.85$** | **$0.9621$** |

### 2. 50% Stenosis Regional Error Decomposition

When trained on Healthy + 30% and evaluated on the unseen 50% stenosis across anatomical regions:

| Anatomical Region | Axial Interval $z$ ($\text{mm}$) | Geometry-Only RMSE | Physics-Guided RMSE | Error Reduction ($\%$) |
| :--- | :---: | :---: | :---: | :---: |
| **Upstream** | $0 \le z < 50$ | $0.88\text{ mm/s}$ | **$0.74\text{ mm/s}$** | $-15.7\%$ |
| **Approach** | $50 \le z < 55$ | $14.12\text{ mm/s}$ | **$13.13\text{ mm/s}$** | $-7.0\%$ |
| **Stenosis Throat** | $55 \le z < 65$ | $131.52\text{ mm/s}$ | **$37.55\text{ mm/s}$** | **$-71.4\%$** |
| **Post-stenotic** | $65 \le z < 85$ | $37.94\text{ mm/s}$ | **$37.54\text{ mm/s}$** | $-1.1\%$ |
| **Distal Recovery**| $85 \le z \le 121$| $3.86\text{ mm/s}$ | **$3.84\text{ mm/s}$** | $-0.5\%$ |

> **Key Finding**: In the constriction throat, standard machine learning incurs severe errors ($131.5\text{ mm/s}$ RMSE) due to out-of-distribution geometric contraction. Incorporating the analytical continuity baseline reduces throat error by **$71.4\%$**, ensuring physical boundedness.

---

## 6. How to Reproduce

### Option A: One-Line Master Reproduction
```bash
./reproduce.sh
```

### Option B: Step-by-Step Python Commands
```bash
# 1. Run automated test suite
pytest tests/ -v

# 2. Re-train models and run spatial holdout experiment
python scripts/train_hemodyn_ml.py

# 3. Test cross-geometry holdout generalization
python scripts/test_hemodyn_geometry_holdout.py

# 4. Perform regional error decomposition on 50% stenosis
python scripts/analyze_hemodyn_ml_regions.py

# 5. Evaluate post-stenotic reverse-flow diagnostic
python scripts/plot_hemodyn_ml_reverse_flow.py

# 6. Render poster summary panel
python scripts/create_ml_poster_panel.py
```

