import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Everything, Loglogistic_Distribution
from reliability.Distributions import Loglogistic_Distribution


index_names = ['engine', 'cycle'] 
operational_condition_names = ['altitude',  'mach_nr', 'TRA']
sensor_names = [
    'T2',        # total temperature at fan inlet
    'T24',       # total temperature at LPC outlet
    'T30',       # total temperature at HPC outlet
    'T50',       # total temperature at LPT outlet
    'P2',        # pressure at fan inlet
    'P15',       # total pressure in bypass-duct
    'P30',       # total pressure at HPC outlet
    'Nf',        # physical fan speed (rpm)
    'Nc',        # physical core speed (rpm)
    'epr',       # engine pressure ratio (P50/P2)
    'Ps30',      # static pressure at HPC outlet
    'phi',       # ratio of fuel flow to Ps30
    'NRf',       # corrected fan speed
    'NRc',       # corrected core speed
    'BPR',       # bypass ratio
    'farB',      # burner fuel-air ratio
    'htBleed',   # bleed enthalpy
    'Nf_dmd',    # demanded fan speed (rpm)
    'PCNfR_dmd', # demanded corrected fan speed (rpm)
    'W31',       # HPT coolant bleed (lbm/s)
    'W32',       # LPT coolant bleed
]

col_names =  index_names + operational_condition_names + sensor_names

df_train = pd.read_csv(r'train_FD001.txt' ,  sep = ' ' , names=col_names, index_col = False,  usecols=range(len(col_names))) 





# Extracting first and second column from the train data set with engine number and cycles:
engine_cycle_data = df_train[['engine', 'cycle']]

# Filtering list that only max cycle for each engine is left
lifetimes = list(df_train.groupby('engine')['cycle'].max().values.astype(float))



## Bullet point 1

# Manually removing distributions not used
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



# Printing the 3 goodness-of-fit criteria
gof_table = fit_results.results[['AICc', 'BIC', 'AD']]
print("\n Goodness-of-Fit Table:")
print(gof_table)







## Bullet point 2

# copying best alpha and beta for the Loglogistic_2P distribution
best_alpha = fit_results.Loglogistic_2P_alpha
best_beta = fit_results.Loglogistic_2P_beta
loglogistic_dist = Loglogistic_Distribution(alpha=best_alpha, beta=best_beta)

# Array of cycle values 
cycle_values = np.linspace(1, 400, 500)

# Compute hazard function values
hf_values = loglogistic_dist.HF(cycle_values)

# Plot hazard function
plt.close('all')
plt.plot(cycle_values, hf_values)
plt.xlabel("Engine Cycles")
plt.ylabel("Hazard h(t)")
plt.grid(True)
plt.savefig('Hazard_Function.png', dpi=300, bbox_inches='tight')
plt.show()







## Bullet point 3


# Given values and new t list
cp = 1e4  
cf = 1e5 
t_array = np.arange(1, 400, 1)
g_t_values = []


for t in t_array:
    t_val = float(t)
    R_t = loglogistic_dist.SF(t_val)
    F_t = loglogistic_dist.CDF(t_val)
    
    # Calculate the expected cycle times
    t_integral = list(np.linspace(0, t_val, 500))
    R_integral = loglogistic_dist.SF(t_integral)
    
    # Compute the integral part
    expected_length = np.trapezoid(R_integral, t_integral)
    
    # Compute expected average costs
    g_t = (cp * R_t + cf * F_t) / expected_length
    g_t_values.append(g_t)

# Compute minumum average cost g(t)
g_t_values = np.array(g_t_values)
optimal_idx = np.argmin(g_t_values)
optimal_t = t_array[optimal_idx]
min_cost = g_t_values[optimal_idx]

# Print final values of g(t) and t itself
print(f"Optimal Time (t*): {optimal_t} cycles")
print(f"Minimum Cost g(t*): {min_cost:.2f} euros/cycle")

# Plot the g(t) vs t
plt.close('all')
plt.plot(t_array, g_t_values, label="Expected Cost g(t)")
plt.axvline(optimal_t, color='red', linestyle='--', label=f'Optimal t* = {optimal_t}')
plt.plot(optimal_t, min_cost, 'ro') 
plt.xlabel("Engine Cycles")
plt.ylabel("Expected Cost per Cycle (Euros)")
plt.legend()
plt.grid(True)
plt.savefig('Cost_Optimization.png', dpi=300, bbox_inches='tight')
plt.show()