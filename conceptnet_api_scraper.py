import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from utils.text2uri import standardized_uri

INPUT_PATH = os.path.join(os.getcwd(), "data", "gender_neutral")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "conceptnet_api")

def preprocess(file_path:str) -> list:
    """
    Helper function to pre-process and clean input csv file.
    """
    df = pd.read_csv(file_path)

    col_name = df.columns[0] # take first column only
    df[col_name] = df[col_name].str.lower() # case-folding
    df = df.drop_duplicates() # remove duplicates

    return df[col_name].to_list()    
    

def main(input_folder:str, output_folder:str, edge_limit:int=500) -> None:

    csv_files = [file for file in os.listdir(input_folder) if file.split('.')[-1]=='csv']

    completed_words = [file.split('.')[0] for file in os.listdir(output_folder) if file.split('.')[-1]=='json']
    
    for file in tqdm(csv_files):

        all_words = preprocess(f"{input_folder}/{file}")
        new_words = [word for word in all_words if word not in completed_words]
        elapsed_time = 1.00 # initialise time as 1 second
        keywords = {'start_label': [], 'end_label': [], 'rel_id': [], 'surface_text': [], 'weight': []}

        for word in tqdm(new_words):

            if elapsed_time < 1.00: # ensure that API usage for 1 request is longer than 1 second
                time.sleep(1.00 - elapsed_time)

            start_time = time.time()
            std_uri = standardized_uri(language='en', term=word)
            obj = requests.get(f'http://api.conceptnet.io{std_uri}?offset=0&limit={edge_limit}&language=en').json()            

            with open(f'{output_folder}/{word}.json', "w") as json_file:
                    json.dump(obj['edges'], json_file, indent=4)

            for edge in obj['edges']:

                # Extract keyword infromation
                keywords['start_label'].append(edge['start']['label'])
                keywords['end_label'].append(edge['end']['label'])
                keywords['rel_id'].append(edge['rel']['@id'])
                keywords['surface_text'].append(edge['surfaceText'])
                keywords['weight'].append(edge['weight'])                

            elapsed_time = time.time() - start_time
        
        keywords_df = pd.DataFrame.from_dict(keywords)
        filename = file.split('.')[0]
        keywords_df.to_csv(f'{output_folder}/{filename}.csv', index=False)
    return


if __name__ == "__main__":    
    main(input_folder=INPUT_PATH, output_folder=OUTPUT_PATH)