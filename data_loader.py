from data_processing import DataLoader

loader = DataLoader("Model training for MT\\Dataset\\input5k.csv")
print(loader.get_cleaned_data().head())

loader.save_cleaned_data("Model training for MT\\Dataset\\processed_data.csv", index=False)

