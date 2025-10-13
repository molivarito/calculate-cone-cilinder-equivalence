# Analysis of Acoustic Conical Resonators

This is a desktop tool with a Graphical User Interface (GUI) developed in Python for the analysis and visualization of the acoustic properties of conical resonators. The application allows for the comparison of resonance frequencies and equivalent cylinder lengths calculated through different theoretical models.

---
## Main Features

* **Intuitive Graphical Interface:** Allows for easy modification of the resonator's parameters and instant visualization of the results.
* **Resonance Calculation:** Implements two models to calculate resonance frequencies:
    * An ideal model.
    * A more robust model that includes the radiation impedance at the cone's ends.
* **Equivalent Length (`L_eq`) Analysis:** Calculates the length of an equivalent cylinder using two theoretical approximations:
    * Perturbation Theory.
    * Spherical Wave Approximation.
* **Comprehensive Data Visualization:** The results are presented in four distinct tabs:
    1.  **Results Table:** Detailed numerical data for each resonance mode.
    2.  **Frequency Analysis:** Plots of `L_eq` and the relative error of each model as a function of frequency.
    3.  **Geometry by Mode:** A detailed view that overlays the physical cone and the equivalent cylinders for a selected mode.
    4.  **Comparative Visualization:** Compares the geometry of the physical cone with the equivalent lengths of all calculated modes.
* **Customizable Parameters:** The user can define the cone's length, initial and final radii, the speed of sound, and the number of modes to calculate.
* **Plot Export:** Each plot can be saved in high-quality formats such as **PDF, PNG, and SVG**, ideal for academic publications.
* **Publication Style:** The plots are generated with a pre-configured style to meet the standards of scientific publications.

---
## Application Views

Below are examples of the different views offered by the tool.

**Main View and Controls**

The main window with parameter controls and action buttons.

*(Image of the application's main window here)*

**Tab: Results Table**

Displays the numerical values of the calculated frequencies and equivalent lengths.

*(Image of the results table tab here)*

**Tab: L vs. Frequency Analysis**

Plots showing the equivalent length and the models' error as a function of frequency.

*(Image of the frequency analysis tab here)*

---
## Theoretical Models Used

The application is based on the following acoustic models:

1.  **Reference Frequencies (`f_ref`)**
    * **Ideal Model:** Calculates frequencies without considering radiation losses at the ends.
    * **Model with Radiation:** Numerically solves a more complex equation that incorporates the radiation impedance at the cone's open ends. This method uses `scipy.optimize.minimize_scalar` to find the wavenumbers (`k`) that minimize the determinant of the system matrix, which corresponds to the resonances.

2.  **Equivalent Length (`L_eq`)**
    Once the reference frequencies are obtained, the lengths of an "equivalent" cylinder (with an average radius `r_avg`) that would resonate at those same frequencies are calculated using two approximations:
    * **Perturbation Theory:** A model that approximates the effect of the conicity as a small perturbation on a cylinder.
    * **Spherical Wave Approximation:** A model that considers the acoustic field inside the cone as a section of a spherical wave.

The objective is to validate how well these `L_eq` approximations match the reference equivalent length, which is calculated directly from the resonance frequencies (`L_eq_ref = n*c / (2*f_ref)`).

---
## Requirements

* Python 3.x
* Tkinter (usually included in the standard Python installation)
* NumPy
* SciPy
* Matplotlib

You can install the necessary dependencies with pip:
```bash
pip install numpy scipy matplotlib