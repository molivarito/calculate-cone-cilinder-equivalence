import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

# --- Global Plot Style for Academic Publications (FINELY TUNED) ---
mpl.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.dpi': 150,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.color': 'lightgray'
})

# --- Physical Parameters ---
AIR_DENSITY = 1.225 # kg/m^3

# --- CALCULATION LOGIC (Unchanged) ---
def calculate_ideal_resonances(L, epsilon, r_avg, c, n_max):
    modes = np.arange(1, n_max + 1)
    k_squared = (modes * np.pi / L)**2 + (epsilon / r_avg)**2
    k_n = np.sqrt(k_squared)
    frequencies_n = (c * k_n) / (2 * np.pi)
    return modes, frequencies_n
def z_radiation(k, radius, c, rho):
    if radius <= 1e-9: return complex(np.inf, np.inf)
    S = np.pi * radius**2
    R_rad = rho * c / S * (k**2 * radius**2) / 2
    X_rad = rho * c / S * (8 * k * radius) / (3 * np.pi)
    return complex(R_rad, X_rad)
def robust_resonance_equation(k, L, r1, r2, epsilon, c, rho):
    if k <= 1e-3: return 1e12
    abs_epsilon = np.abs(epsilon) if epsilon != 0 else 1e-9
    r_a, r_b = r1 / abs_epsilon, r2 / abs_epsilon
    Z_rad_a = z_radiation(k, r1, c, rho); Z_rad_b = z_radiation(k, r2, c, rho)
    C11 = np.sin(k*r_a)/r_a - Z_rad_a/(1j*k*c*rho) * (k*np.cos(k*r_a)/r_a - np.sin(k*r_a)/r_a**2)
    C12 = np.cos(k*r_a)/r_a - Z_rad_a/(1j*k*c*rho) * (-k*np.sin(k*r_a)/r_a - np.cos(k*r_a)/r_a**2)
    C21 = np.sin(k*r_b)/r_b + Z_rad_b/(1j*k*c*rho) * (k*np.cos(k*r_b)/r_b - np.sin(k*r_b)/r_b**2)
    C22 = np.cos(k*r_b)/r_b + Z_rad_b/(1j*k*c*rho) * (-k*np.sin(k*r_b)/r_b - np.cos(k*r_b)/r_b**2)
    determinant = C11 * C22 - C12 * C21
    return np.abs(determinant)
def calculate_resonances_with_radiation(L, r1, r2, epsilon, c, rho, n_max, debug_prints=False):
    modes = np.arange(1, n_max + 1)
    k_max = (n_max + 2) * np.pi / L
    k_scan = np.linspace(1e-2, k_max, 4000)
    v_resonance_eq = np.vectorize(robust_resonance_equation)
    scan_values = v_resonance_eq(k_scan, L, r1, r2, epsilon, c, rho)
    minima_indices, _ = find_peaks(-scan_values, distance=100)
    if debug_prints:
        print("\n--- DEBUG START: Searching for Resonances with Radiation ---")
        print(f"Scanned 'k' range: {k_scan[0]:.2f} to {k_scan[-1]:.2f}")
        print(f"Found {len(minima_indices)} local minima at approx k: {[f'{k_scan[i]:.2f}' for i in minima_indices]}")
    k_resonances = []
    for i, idx in enumerate(minima_indices):
        if i >= n_max: break
        k_approx = k_scan[idx]
        bracket_low = k_scan[max(0, idx - 10)]; bracket_high = k_scan[min(len(k_scan)-1, idx + 10)]
        res = minimize_scalar(robust_resonance_equation, args=(L, r1, r2, epsilon, c, rho), bracket=(bracket_low, bracket_high), method='brent')
        if res.success:
            k_n = res.x
            k_resonances.append(k_n)
            if debug_prints: print(f"  Mode {i+1}: k_approx={k_approx:.3f} -> Fine optimization -> k_res={k_n:.5f} (Freq: {c*k_n/(2*np.pi):.2f} Hz)")
        else:
            k_resonances.append(np.nan)
            if debug_prints: print(f"  Mode {i+1}: Optimization failed near k={k_approx:.3f}")
    if debug_prints: print("--- DEBUG END ---\n")
    while len(k_resonances) < n_max: k_resonances.append(np.nan)
    frequencies_n = (c * np.array(k_resonances)) / (2 * np.pi)
    return modes, frequencies_n
def calculate_equivalent_lengths(modes, freqs_ref, L, epsilon, r_avg, c):
    leq_ref = modes * c / (2 * freqs_ref)
    omegas_ref = 2 * np.pi * freqs_ref; omegas_ref[np.isnan(omegas_ref)] = 1e-9; omegas_ref[omegas_ref == 0] = 1e-9
    term_pert = 0.5 * (epsilon * c / (omegas_ref * r_avg))**2
    leq_pert = L * (1 - term_pert)
    term_spher = (2 * c**2 * np.abs(epsilon)) / (omegas_ref**2 * L * r_avg)
    leq_spher = L * (1 + term_spher)
    return leq_ref, leq_pert, leq_spher

# --- MAIN GUI APPLICATION CLASS ---
class AcousticConeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Conical Resonator Analysis")
        self.geometry("1100x850")
        self.datos = {}
        self._create_controls()
        self._create_notebook()
        self.update_all()

    def _create_controls(self):
        controls_frame = ttk.Frame(self, padding="10")
        controls_frame.pack(side="top", fill="x", expand=False)
        self.params = { "Length L (m):": "1.0", "Initial Radius r1 (m):": "0.05", "Final Radius r2 (m):": "0.03", "Sound Speed c (m/s):": "343.0", "Number of Modes:": "15" }
        self.entries = {}
        params_frame = ttk.LabelFrame(controls_frame, text="Resonator Parameters")
        params_frame.pack(side="left", fill="x", expand=True, padx=5)
        for i, (label, value) in enumerate(self.params.items()):
            ttk.Label(params_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            entry = ttk.Entry(params_frame, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            entry.insert(0, value); self.entries[label] = entry
        model_frame = ttk.LabelFrame(controls_frame, text="Model Configuration")
        model_frame.pack(side="left", fill="x", expand=True, padx=5)
        self.include_radiation = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_frame, text="Include Radiation Impedance", variable=self.include_radiation).pack(pady=5, padx=10, anchor="w")
        self.debug_prints = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_frame, text="Enable Debug Output", variable=self.debug_prints).pack(pady=5, padx=10, anchor="w")
        update_button = ttk.Button(controls_frame, text="Calculate &\nUpdate", command=self.update_all, style="Accent.TButton")
        update_button.pack(side="left", fill="both", expand=True, padx=10)
        ttk.Style().configure("Accent.TButton", font=("Helvetica", 10, "bold"))

    def _create_notebook(self):
        notebook = ttk.Notebook(self)
        notebook.pack(side="bottom", fill="both", expand=True)
        self.tab_table = ttk.Frame(notebook, padding="10"); self.tab_geo_mode = ttk.Frame(notebook, padding="10")
        self.tab_analysis_freq = ttk.Frame(notebook, padding="10"); self.tab_geo_multi = ttk.Frame(notebook, padding="10")
        notebook.add(self.tab_table, text="Results Table")
        notebook.add(self.tab_geo_mode, text="Geometry by Mode")
        notebook.add(self.tab_analysis_freq, text="L vs. Frequency Analysis")
        notebook.add(self.tab_geo_multi, text="Comparative Visualization")
        self._setup_table_tab(); self._setup_geo_mode_tab(); self._setup_analysis_freq_tab(); self._setup_geo_multi_tab()

    def _add_save_button(self, parent_frame, figure_object):
        """Helper function to add a save button to a tab."""
        save_button_frame = ttk.Frame(parent_frame)
        save_button_frame.pack(side="bottom", fill="x", pady=(5,0))
        save_button = ttk.Button(save_button_frame, text="Save Plot", 
                                 command=lambda: self.save_plot(figure_object))
        save_button.pack()

    def _setup_table_tab(self):
        columns = ("n", "f_ref", "leq_ref", "leq_pert", "err_pert", "leq_spher", "err_spher")
        self.tree = ttk.Treeview(self.tab_table, columns=columns, show="headings")
        headings = { "n": "Mode n", "f_ref": "F_ref (Hz)", "leq_ref": "L_eq Ref (m)", "leq_pert": "L_eq Pert. (m)", "err_pert": "Error Pert. (%)", "leq_spher": "L_eq Sph. (m)", "err_spher": "Error Sph. (%)" }
        for col, heading_text in headings.items(): self.tree.heading(col, text=heading_text); self.tree.column(col, width=120, anchor="center")
        scrollbar = ttk.Scrollbar(self.tab_table, orient="vertical", command=self.tree.yview); self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        
    def _setup_analysis_freq_tab(self):
        self.fig_analysis, self.axs_analysis = plt.subplots(2, 1, sharex=True)
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, master=self.tab_analysis_freq)
        self.canvas_analysis.get_tk_widget().pack(fill="both", expand=True)
        self._add_save_button(self.tab_analysis_freq, self.fig_analysis)

    def _setup_geo_multi_tab(self):
        self.fig_geo_multi, self.axs_geo_multi = plt.subplots(2, 1, sharex=True)
        self.canvas_geo_multi = FigureCanvasTkAgg(self.fig_geo_multi, master=self.tab_geo_multi)
        self.canvas_geo_multi.get_tk_widget().pack(fill="both", expand=True)
        self._add_save_button(self.tab_geo_multi, self.fig_geo_multi)
        
    def _setup_geo_mode_tab(self):
        frame = self.tab_geo_mode
        control_frame = ttk.Frame(frame); control_frame.pack(side="top", fill="x", pady=5)
        ttk.Label(control_frame, text="Select Mode to Display:").pack(side="left", padx=5)
        self.mode_selection = tk.StringVar()
        self.mode_combobox = ttk.Combobox(control_frame, textvariable=self.mode_selection, state="readonly")
        self.mode_combobox.pack(side="left", padx=5)
        self.mode_combobox.bind("<<ComboboxSelected>>", self.update_single_mode_geo_plot)
        
        self.fig_single_mode, self.ax_single_mode = plt.subplots()
        self.canvas_single_mode = FigureCanvasTkAgg(self.fig_single_mode, master=frame)
        self.canvas_single_mode.get_tk_widget().pack(fill="both", expand=True)
        self._add_save_button(frame, self.fig_single_mode)

    def update_all(self):
        try:
            L = float(self.entries["Length L (m):"].get()); r1 = float(self.entries["Initial Radius r1 (m):"].get())
            r2 = float(self.entries["Final Radius r2 (m):"].get()); c = float(self.entries["Sound Speed c (m/s):"].get())
            n_max = int(self.entries["Number of Modes:"].get())
            if L <= 0 or r1 <= 0 or r2 <= 0 or c <= 0 or n_max < 1: raise ValueError("Parameters must be positive.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numerical values.\n{e}"); return
        epsilon = (r2 - r1) / L; r_avg = (r1 + r2) / 2
        if self.include_radiation.get():
            print("Calculating with radiation impedance (robust method)...")
            modes, freqs_ref = calculate_resonances_with_radiation(L, r1, r2, epsilon, c, AIR_DENSITY, n_max, self.debug_prints.get())
        else:
            print("Calculating with ideal model (no radiation)...")
            modes, freqs_ref = calculate_ideal_resonances(L, epsilon, r_avg, c, n_max)
        leq_ref, leq_pert, leq_spher = calculate_equivalent_lengths(modes, freqs_ref, L, epsilon, r_avg, c)
        self.datos = { 'modes': modes, 'frecs_ref': freqs_ref, 'leq_ref': leq_ref, 'leq_pert': leq_pert, 'leq_spher': leq_spher, 'L': L, 'r1': r1, 'r2': r2, 'epsilon': epsilon, 'r_avg': r_avg }
        self.update_table(); self.update_analysis_plot(); self.update_multi_geo_plot(); self.update_single_mode_controls()
        print("Calculation complete.")

    def update_table(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        d = self.datos
        with np.errstate(divide='ignore', invalid='ignore'):
            error_pert = 100 * (d['leq_pert'] - d['leq_ref']) / d['leq_ref']
            error_spher = 100 * (d['leq_spher'] - d['leq_ref']) / d['leq_ref']
        for i, mode in enumerate(d['modes']):
            row_data = ( mode, f"{d['frecs_ref'][i]:.2f}", f"{d['leq_ref'][i]:.5f}", f"{d['leq_pert'][i]:.5f}", f"{error_pert[i]:.5f}", f"{d['leq_spher'][i]:.5f}", f"{error_spher[i]:.5f}" )
            self.tree.insert("", "end", values=row_data)

    def update_analysis_plot(self):
        ax1, ax2 = self.axs_analysis; ax1.clear(); ax2.clear()
        d = self.datos
        with np.errstate(divide='ignore', invalid='ignore'):
            error_pert = 100 * (d['leq_pert'] - d['leq_ref']) / d['leq_ref']; error_spher = 100 * (d['leq_spher'] - d['leq_ref']) / d['leq_ref']
        ax1.axhline(d['L'], color='black', linestyle=':', linewidth=1, label=f"Physical Length ({d['L']:.3f} m)")
        ax1.plot(d['frecs_ref'], d['leq_ref'], 'ko', mfc='white', label=r'Reference $L_{eq}$')
        ax1.plot(d['frecs_ref'], d['leq_pert'], 's', color='C0', mfc='white', label=r'Perturbation $L_{eq}$ Approx.')
        ax1.plot(d['frecs_ref'], d['leq_spher'], '^', color='C1', mfc='white', label=r'Spherical Wave $L_{eq}$ Approx.')
        ax1.set_ylabel("Equivalent Length (m)"); ax1.set_title("Equivalent Length vs. Resonance Frequency")
        ax1.legend(); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax2.plot(d['frecs_ref'], error_pert, '-s', color='C0', mfc='white', label='Error: Perturbation Approx.')
        ax2.plot(d['frecs_ref'], error_spher, '-^', color='C1', mfc='white', label='Error: Spherical Wave Approx.')
        ax2.axhline(0, color='black', linestyle=':', linewidth=1)
        ax2.set_xlabel("Resonance Frequency (Hz)"); ax2.set_ylabel("Relative Error in $L_{eq}$ (%)")
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False); ax2.legend()
        self.fig_analysis.suptitle(r"Analysis of Equivalent Length Models, $L_{eq}(\omega)$")
        self.fig_analysis.tight_layout(rect=[0, 0, 1, 0.96]); self.canvas_analysis.draw()

    def update_multi_geo_plot(self):
        ax1, ax2 = self.axs_geo_multi; ax1.clear(); ax2.clear()
        d = self.datos
        ax1.set_title("1. Physical Cone Geometry")
        x_cone = np.array([0, d['L']]); y_cone = d['r1'] + d['epsilon'] * x_cone
        ax1.plot(x_cone, y_cone, 'k-', lw=1.5); ax1.plot(x_cone, -y_cone, 'k-', lw=1.5)
        ax1.fill_between(x_cone, y_cone, -y_cone, color='black', alpha=0.1)
        ax1.axvline(d['L'], color='black', ls='--', lw=1, label=f"Physical Length ({d['L']:.3f} m)")
        ax1.set_ylabel("Radius (m)"); ax1.legend(); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax2.set_title("2. Comparison of Equivalent Lengths ($L_{eq}$) for each Mode")
        y_pos = np.arange(len(d['modes']), 0, -1); bar_height = 0.35
        ax2.barh(y_pos + bar_height/2, d['leq_pert'], height=bar_height, color='w', ec='C0', hatch='//', label='Perturbation Model')
        ax2.barh(y_pos - bar_height/2, d['leq_spher'], height=bar_height, color='w', ec='C1', hatch='\\\\', label='Spherical Wave Model')
        ax2.axvline(d['L'], color='black', ls='--', lw=1)
        ax2.set_yticks(y_pos); ax2.set_yticklabels([f"n={n}" for n in d['modes']])
        ax2.invert_yaxis()
        ax2.set_xlabel("Equivalent Length (m)"); ax2.legend()
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        self.fig_geo_multi.suptitle("Comparative Geometry Visualization")
        self.fig_geo_multi.tight_layout(rect=[0, 0, 1, 0.96]); self.canvas_geo_multi.draw()

    def update_single_mode_controls(self):
        available_modes = list(self.datos['modes'])
        self.mode_combobox['values'] = available_modes
        if available_modes: self.mode_selection.set(available_modes[0]); self.update_single_mode_geo_plot()

    def update_single_mode_geo_plot(self, event=None):
        ax = self.ax_single_mode; ax.clear()
        if not self.datos or not self.mode_selection.get(): self.canvas_single_mode.draw(); return
        mode_n = int(self.mode_selection.get()); idx = mode_n - 1; d = self.datos
        l_pert = d['leq_pert'][idx]; l_spher = d['leq_spher'][idx]
        x_cone = np.array([0, d['L']]); y_cone = d['r1'] + d['epsilon'] * x_cone
        ax.plot(x_cone, y_cone, 'k-', lw=1.5); ax.plot(x_cone, -y_cone, 'k-', lw=1.5)
        ax.fill_between(x_cone, y_cone, -y_cone, color='black', alpha=0.1)
        ax.add_patch(plt.Rectangle((0, -d['r_avg']), l_pert, 2*d['r_avg'], ec='C0', fc='none', ls='--', lw=1.5))
        ax.add_patch(plt.Rectangle((0, -d['r_avg']), l_spher, 2*d['r_avg'], ec='C1', fc='none', ls=':', lw=1.5))
        ax.set_title(f"Geometry Visualization for Mode n = {mode_n}")
        ax.set_xlabel("Length (m)"); ax.set_ylabel("Radius (m)")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.annotate(f"Physical Cone (L={d['L']:.3f} m)", xy=(d['L']/2, y_cone[0]/2), xytext=(d['L']/2, max(d['r1'],d['r2'])*1.5), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"))
        ax.annotate(f"Perturbation Eq. Cyl. (L={l_pert:.5f} m)", xy=(l_pert/2, -d['r_avg']), xytext=(l_pert/2, -d['r_avg']*4), ha='center', color='C0', arrowprops=dict(arrowstyle="->", color='C0', connectionstyle="arc3,rad=-0.2"))
        ax.annotate(f"Sph. Wave Eq. Cyl. (L={l_spher:.5f} m)", xy=(l_spher/1.5, d['r_avg']), xytext=(l_spher/1.5, d['r_avg']*4), ha='center', color='C1', arrowprops=dict(arrowstyle="->", color='C1', connectionstyle="arc3,rad=0.2"))
        max_x = max(d['L'], l_pert, l_spher); max_y = max(d['r1'], d['r2'])
        ax.set_xlim(-0.1*max_x, max_x*1.1); ax.set_ylim(-max_y*4.5, max_y*4.5)
        self.fig_single_mode.tight_layout(); self.canvas_single_mode.draw()

    def save_plot(self, figure_object):
        """Opens a dialog to save the given figure object."""
        file_path = filedialog.asksaveasfilename(
            parent=self, title="Save Plot As",
            filetypes=[("PDF file", "*.pdf"), ("PNG file", "*.png"), ("SVG file", "*.svg"), ("All files", "*.*")],
            defaultextension=".pdf"
        )
        if file_path:
            try:
                figure_object.savefig(file_path, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error Saving File", f"An error occurred while saving the plot:\n{e}")

if __name__ == "__main__":
    app = AcousticConeGUI()
    app.mainloop()