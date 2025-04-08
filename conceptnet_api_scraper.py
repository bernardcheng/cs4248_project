import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from utils.text2uri import standardized_uri
from scraper_filter import create_default_filter

KEYWORD_PATH = os.path.join(os.getcwd(), "data", "gender_neutral")
JSON_PATH = os.path.join(os.getcwd(), "data", "conceptnet_api", "json")
CSV_PATH = os.path.join(os.getcwd(), "data", "conceptnet_api", "csv")

# Default filter configuration
DEFAULT_FILTER = create_default_filter()

def preprocess(file_path:str) -> list:
    """
    Helper function to pre-process and clean input csv file.
    """
    df = pd.read_csv(file_path)

    col_name = df.columns[0] # take first column only
    df[col_name] = df[col_name].str.lower() # case-folding
    df = df.drop_duplicates() # remove duplicates

    return df[col_name].to_list()    
    

def scrape_keywords(input_folder:str, output_folder:str, edge_limit:int=500) -> None:
    """
    Sends URI keyword request to ConceptNet API Server to obtain and store raw query results in JSON format.
    """

    csv_files = [file for file in os.listdir(input_folder) if file.split('.')[-1]=='csv']

    completed_words = [file.split('.')[0] for file in os.listdir(output_folder) if file.split('.')[-1]=='json']
    
    for file in tqdm(csv_files):

        all_words = preprocess(f"{input_folder}/{file}")
        new_words = [word for word in all_words if word not in completed_words]
        elapsed_time = 1.00 # initialise time as 1 second
        

        for word in tqdm(new_words):

            if elapsed_time < 1.00: # ensure that API usage for 1 request is longer than 1 second
                time.sleep(1.00 - elapsed_time)

            start_time = time.time()
            std_uri = standardized_uri(language='en', term=word)
            obj = requests.get(f'http://api.conceptnet.io/query?node={std_uri}&other=/c/en&limit={edge_limit}').json()     

            with open(f'{output_folder}/{word}.json', "w") as json_file:
                    json.dump(obj['edges'], json_file, indent=4)     

            elapsed_time = time.time() - start_time 
    return

def parse_response(input_folder:str, output_folder:str, edge_filter=DEFAULT_FILTER) -> None:
    """
    Helper function to read raw JSON files and extract key field information from each edge.        
    """
    json_files = [file for file in os.listdir(input_folder) if file.split('.')[-1]=='json']

    keywords = {'end_id': [], 'end_label': [], 'start_id': [], 'start_label': [], 'rel_id': [], 'surface_text': [], 'weight': [], 'dataset': []}

    for file in tqdm(json_files):
        with open(f"{input_folder}/{file}", 'r') as f:
            data = json.load(f)
            
            # Apply the filter
            filtered_edges = edge_filter(data)

            # Store filtered edge information
            for edge in filtered_edges:
                keywords['end_id'].append(edge['end']['@id'])
                keywords['end_label'].append(edge['end']['label'])
                keywords['start_id'].append(edge['start']['@id'])
                keywords['start_label'].append(edge['start']['label'])                
                keywords['rel_id'].append(edge['rel']['@id'])
                keywords['surface_text'].append(edge['surfaceText'])
                keywords['weight'].append(edge['weight']) 
                keywords['dataset'].append(edge['dataset'])   

    keywords_df = pd.DataFrame.from_dict(keywords).drop_duplicates()
    keywords_df.to_csv(f'{output_folder}/edge_extract.csv', index=False)
    return


if __name__ == "__main__":    
    # scrape_keywords(input_folder=KEYWORD_PATH, output_folder=JSON_PATH)
    from scraper_filter import (
        create_basic_weight_filter,
        create_average_weighted_relation_filter,
        create_high_quality_filter,
        create_multilingual_weighted_filter,
        create_relation_specific_filter,
        create_dataset_specific_filter,
        create_comprehensive_filter,
        create_statistical_outlier_filter,
        create_semantic_similarity_filter
    )

    parse_response(input_folder=JSON_PATH, output_folder=CSV_PATH)