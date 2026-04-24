import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self._data_load_and_clean()

    def _data_load_and_clean(self):
        # load raw csv file
        df = pd.read_csv(self.filepath)

        # Fill empty values in final_text_version using text_data
        df["final_text_data"] = df["final_text_data"].replace("", pd.NA)
        df["source_text"] = df["final_text_data"].fillna(df["text_data"])

        # keep desired columns
        desired_cols = ["source_text", "final_text_version", "source_language", "target_language"]
        cols_to_drop = [col for col in df.columns if col not in desired_cols]
        df = df.drop(columns=cols_to_drop)

        # Rename target column
        df = df.rename(columns={"final_text_version": "target_text"})

        self.df = df

    def get_cleaned_data(self):
        return self.df.copy()
    
    def save_cleaned_data(self, output_path: str, index: bool = False):
        self.df.to_csv(output_path, index=index)





