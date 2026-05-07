from data_processing import DataLoader

loader = DataLoader("MT-Model-Taining-Pipeline/Dataset/(AcceptedMachineTranslationData-2026-04-22 - (AcceptedMachineTranslationData-2026-04-22.csv")
print(loader.get_cleaned_data().head())
loader.save_cleaned_data("/home/mushrat/MT-model-training-pipeline/MT-Model-Taining-Pipeline/output/processed_data.csv", index=False)

