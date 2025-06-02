# %%
import pandas as pd
# %%
df = pd.read_csv("/home/users/u12559743/DAVI/IC/Codes/data_aug.csv")
print(len(df))
df_subset = df[['PAN Path', 'Sex']]
df_subset.to_csv('Patient_Sex.csv', index = False)
# %%
df_subset['PAN Path'][1]
# %%
df_subset['PAN Path'] = df_subset['PAN Path'].str.replace("/home/12559743", "/home/users/u12559743")
# %%
df_subset
# %%
df_subset.to_csv('Patient_Sex.csv', index = False)
# %%
