import requests
import pandas as pd
from utils.text2uri import standardized_uri

def preprocess(file_path:str) -> list:
    """
    Helper function to pre-process and clean input csv file.
    """
    df = pd.read_csv(file_path)

    col = df.columns[0]
    df[col] = df[col].str.lower()

    df = df.drop_duplicates()

    return df[col].to_list()    
    

def main(file_path:str) -> None:
 
    word_list = preprocess(file_path)
    
    pass


if __name__ == "__main__":
    pass