from data_processing import DataLoader

loader = DataLoader("D:\\office\\EBLICT\\Model training for MT\\(AcceptedMachineTranslationData-2026-04-22.csv")
print(loader.get_cleaned_data().head())

loader.save_cleaned_data("D:\\office\\EBLICT\\Model training for MT\\output_clean.csv", index=False)

