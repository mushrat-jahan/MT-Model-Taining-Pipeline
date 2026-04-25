from data_processing import DataLoader

loader = DataLoader("D:\\office\\EBLICT\\Model training for MT\\Dataset\\input5k.csv")
print(loader.get_cleaned_data().head())

loader.save_cleaned_data("D:\\office\\EBLICT\\Model training for MT\\Dataset\\processed_data.csv", index=False)

