import os

from conceptnet_api_scraper import parse_response
import filter_ablation

import pandas as pd

JSON_PATH = os.path.join(os.getcwd(), "data", "conceptnet_api", "json")
CSV_PATH = os.path.join(os.getcwd(), "data", "conceptnet_api", "filtered", "csv")

for filter_name, filter in filter_ablation.get_all_filter_chains().items(): 
  keywords_df = parse_response(input_folder=JSON_PATH, output_folder=CSV_PATH, edge_filter=filter)
  keywords_df.to_csv(f'data/conceptnet_api/csv/edge_extract_{filter_name}.csv', index=False)