# from data_processing import DataLoader
from test import DataLoader

loader = DataLoader("D:\\office\\EBLICT\\Model training for MT\\Untitled spreadsheet - Sheet1.csv")
print(loader.get_cleaned_data().head())

loader.save_cleaned_data("D:\\office\\EBLICT\\Model training for MT\\output_clean11.csv", index=False)

