
"""
Code for reading in the C-MAPSS data (FD001) 
"""
import pandas as pd

"""
For your final RUL prediction model (part 2 of the assignment), do NOT use the engine number or cycle number as input for your model.
The lifetimes of the engine in this simulated dataset is relatively short compared to the life time of real-life engines. 
The cycle number therefore gives more information about the RUL than would be the case in real life.
You do need the cycle number and engine number to calculate the RUL. 
"""
index_names = ['engine', 'cycle'] 

# Operational conditions: altitude, mach_number and throttle_resolver_angle
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

df_train = pd.read_csv(r'C:\Users\joche\OneDrive\Documenten\Master FPP\MMA (Maintenance Modelling & Analysis)\Assignment\6. Turbofan Engine Degradation Simulation Data Set\train_FD001.txt' ,  sep = ' ' , names=col_names, index_col = False,  usecols=range(len(col_names))) 

df_test = pd.read_csv(r'C:\Users\joche\OneDrive\Documenten\Master FPP\MMA (Maintenance Modelling & Analysis)\Assignment\6. Turbofan Engine Degradation Simulation Data Set\test_FD001.txt' , sep=' ' , names= col_names, index_col = False,  usecols=range(len(col_names)))



