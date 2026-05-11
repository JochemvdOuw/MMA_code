
import pandas as pd



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

#C:\Users\joche\OneDrive\Documenten\Master FPP\MMA (Maintenance Modelling & Analysis)\Assignment\6. Turbofan Engine Degradation Simulation Data Set\

# Extracting the 0th ('engine') and 1st ('cycle') columns:
engine_cycle_data = df_train[['engine', 'cycle']]

# To get "how many flight cycles it has been used" before failure for Part 1:
# We group by the engine unit and extract the maximum cycle value.
lifetimes = df_train.groupby('engine')['cycle'].max().values

print(engine_cycle_data)
print(lifetimes)







from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P
import matplotlib.pyplot as plt

# Precondition: 'lifetimes' is a 1-dimensional array of positive numerical values.

# Fit a 2-Parameter Weibull distribution
weibull_model = Fit_Weibull_2P(failures=lifetimes, show_probability_plot=True)

# Fit a 2-Parameter Lognormal distribution
lognormal_model = Fit_Lognormal_2P(failures=lifetimes, show_probability_plot=True)

plt.show()

# Access the optimized parameters mathematically
print("Weibull Shape (Beta):", weibull_model.beta)
print("Weibull Scale (Alpha):", weibull_model.alpha)






from reliability.Fitters import Fit_Everything
import matplotlib.pyplot as plt

# Fit all available models to the failure data
# The function automatically excludes distributions not applicable to the data range
comprehensive_fit = Fit_Everything(failures=lifetimes)
plt.show()

# Extract the DataFrame ranking the distributions by best fit (lowest BIC)
print(comprehensive_fit.results)







from reliability.Distributions import Weibull_Distribution
import matplotlib.pyplot as plt

# Instantiate the mathematical object with known parameters
dist = Weibull_Distribution(alpha=150, beta=2.5)

# Calculate the exact value of the Hazard Function at a specific cycle t
t = 120
hazard_value = dist.HF(t)
print(f"Hazard rate at t={t}: {hazard_value}")

# Visually extract the shape of the hazard function
dist.plot()
plt.show()









import pandas as pd
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Everything

# 1. Define columns as done previously
index_names = ['engine', 'cycle'] 
operational_condition_names = ['altitude', 'mach_nr', 'TRA']
sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 
                'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 
                'PCNfR_dmd', 'W31', 'W32']
col_names = index_names + operational_condition_names + sensor_names

# 2. Read the data
df_train = pd.read_csv(
    r'train_FD001.txt', # Make sure to use your local path here
    sep='\s+', 
    names=col_names, 
    index_col=False
) 

# 3. Extract the maximum cycle for each engine (Time-to-failure)
lifetimes = df_train.groupby('engine')['cycle'].max().values

# 4. Use the reliability package to fit all distributions
# This will automatically print the ranked results (AIC/BIC) and plot the curves
fit_results = Fit_Everything(failures=lifetimes, show_histogram=True)

# 5. Extract the rankings to a CSV if you want to include a table in your report
results_table = fit_results.results
results_table.to_csv('Distribution_Fits_Ranked.csv')

plt.title("Fitted Probability Distributions on Engine Lifetimes")
plt.show()