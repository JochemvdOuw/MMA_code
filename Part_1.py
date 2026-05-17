import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Everything, Loglogistic_Distribution
from reliability.Distributions import Loglogistic_Distribution


cp = 1e4
cf = 1e5
index_names = ['engine', 'cycle'] 
operational_condition_names = ['altitude',  'mach_nr', 'TRA']
sensor_names = ['T2', # total temperature at fan inlet
                'T24',# total temperature at LPC outlet
                'T30', # total temperature at HPC outlet
                'T50', # total temperature at LPT outlet
                'P2', # Pressure at fan inlet
                'P15', #Total pressure in bypass-duct
                'P30', #Total pressure at HPC outlet
                'Nf', #Physical fan speed rpm
                'Nc', #Physical core speed rpm
                'epr', #Engine pressure ratio (P50/P2)
                'Ps30', #Static pressure at HPC outlet
                'phi', #Ratio offuel flow to Ps30
                'NRf', #Corrected fan speed
                'NRc', #Corrected core speed
                'BPR', #Bypass Ratio
                'farB', #Burner fuel-air ratio
                'htBleed', #Bleed Enthalpy
                'Nf_dmd', # Demanded fan speed rpm
                'PCNfR_dmd', #Demanded corrected fan speed rpm
                'W31', #HPT coolant bleed lbm/s
                'W32', #LPT coolant bleed
                ]
# options to visualize the datadrame
col_names =  index_names + operational_condition_names + sensor_names

df_train = pd.read_csv(r'train_FD001.txt' ,  sep = ' ' , names=col_names, index_col = False,  usecols=range(len(col_names))) 


# Extracting the 0th ('engine') and 1st ('cycle') columns:
engine_cycle_data = df_train[['engine', 'cycle']]

# Extract Time-to-Failure (maximum cycles per engine)
lifetimes = list(df_train.groupby('engine')['cycle'].max().values.astype(float))


# =========================================================
# BULLET 1: FITTING DISTRIBUTIONS
# =========================================================
print("--- 1. Fitting All Distributions ---")


distributions_to_exclude = [
    'Exponential_1P',
    'Exponential_2P',
    'Weibull_CR',
    'Weibull_2P',
    'Weibull_DS',
    'Gumbel_2P',

    'Lognormal_2P',
    'Lognormal_3P',
    'Normal_2P',
    'Gamma_2P',
    'Gamma_3P',
    'Weibull_Mixture',
    'Weibull_3P',
    'loglogistic_3P'
]

fit_results = Fit_Everything(
    failures=lifetimes, 
    exclude=distributions_to_exclude, 
    show_histogram_plot=True,
    show_probability_plot=True, 
    show_PP_plot=True,
    show_best_distribution_probability_plot=True
)


# Apply title to the canvas Fit_Everything just created
#plt.title("Fitted Probability Distributions on Engine Lifetimes")
#plt.savefig('Distribution_Fits.png', dpi=300, bbox_inches='tight')
#plt.show(block=True) 

# Print the Goodness of Fit table
gof_table = fit_results.results[['AICc', 'BIC', 'AD']]
print("\nGoodness-of-Fit Table:")
print(gof_table)








# =========================================================
# BULLET 2: HAZARD FUNCTION OF BEST FIT (Loglogistic 2P)
# =========================================================
print("\n--- 2. Hazard Function Analysis ---")
plt.close('all') # Wipe memory again before the next plot

best_alpha = fit_results.Loglogistic_2P_alpha
best_beta = fit_results.Loglogistic_2P_beta

loglogistic_dist = Loglogistic_Distribution(alpha=best_alpha, beta=best_beta)

# The reliability package will draw the Hazard Function directly
# Create an array of time values (e.g., from cycle 1 to 350)
t_values = np.linspace(1, 400, 500)

# Calculate the Hazard Function values explicitly using the mathematical object
hf_values = loglogistic_dist.HF(t_values)

# Plot using standard matplotlib commands to bypass the internal library error
plt.figure(figsize=(8, 5))
plt.plot(t_values, hf_values, label=rf'Loglogistic_2P HF ($\alpha={best_alpha:.2f}$, $\beta={best_beta:.2f}$)')
#plt.title("Hazard Function of the Fitted Loglogistic_2P Distribution")
plt.xlabel("Engine Cycles")
plt.ylabel("Hazard h(t)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save and display
plt.savefig('Hazard_Function.png', dpi=300, bbox_inches='tight')
plt.show(block=True)













# =========================================================
# BULLET 3: OPTIMAL MAINTENANCE TIME & COST
# =========================================================
print("\n--- 3. Optimal Preventive Maintenance Optimization ---")
plt.close('all') # Wipe memory for the final custom plot

cp = 1e4  
cf = 1e5 
t_array = np.arange(1, 400, 1)
g_t_values = []


for t in t_array:
    # Cast numpy integer to native Python float to satisfy the package's type checker
    t_val = float(t)
    
    R_t = loglogistic_dist.SF(t_val)
    F_t = loglogistic_dist.CDF(t_val)
    
    # Calculate the expected cycle length up to time t
    # Cast the numpy array to a standard Python list
    t_integral = list(np.linspace(0, t_val, 500))
    R_integral = loglogistic_dist.SF(t_integral)
    
    # Calculate the integral using numpy's trapezoidal rule
    expected_length = np.trapezoid(R_integral, t_integral)
    
    # Cost formula
    g_t = (cp * R_t + cf * F_t) / expected_length
    g_t_values.append(g_t)

g_t_values = np.array(g_t_values)
optimal_idx = np.argmin(g_t_values)
optimal_t = t_array[optimal_idx]
min_cost = g_t_values[optimal_idx]

print(f"Optimal Time (t*): {optimal_t} cycles")
print(f"Minimum Cost g(t*): {min_cost:.2f} euros/cycle")

# Manually draw the final cost plot
plt.figure(figsize=(8, 5))
plt.plot(t_array, g_t_values, label="Expected Cost g(t)")
plt.axvline(optimal_t, color='red', linestyle='--', label=f'Optimal t* = {optimal_t}')
plt.plot(optimal_t, min_cost, 'ro') 
#plt.title("Long-Term Average Maintenance Cost vs. Replacement Interval")
plt.xlabel("Engine Cycles")
plt.ylabel("Expected Cost per Cycle (Euros)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Cost_Optimization.png', dpi=300, bbox_inches='tight')
plt.show(block=True)