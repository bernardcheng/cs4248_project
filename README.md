# CS4248 Project
Project Repository for Team 34


### Set-up

Pre-requisities: Python Version: 3.11 or newer

1. Create environment 

    a. Using conda:

    ```bash
    conda env create -f environment.yml
    ```

    b.Using pip:
    ```bash
    pip install -r requirements.txt
    ```

    c. Using `uv`:
    ```bash
    uv sync
    ```



### Export environment

1. Via Conda

    * `--no-builds` unties dependencies from OS and Python version.

    ```bash
    conda env export --no-builds > environment.yml
    ```

2. Via Pip
    ```bash
    pip3 freeze > requirements.txt
    ```
    ```bash
    pip freeze > requirements.txt
    ```

3. Via `uv`
    ```bash
    uv export > environment.yml
    ```

### Data download (Google Drive Folder - [conceptnet_api](https://drive.google.com/drive/folders/1uRUyJ6fJibOaSdrZ3C-0QXYwrtH4Efj7?usp=sharing))

1. ConceptNet Query JSON files

2. ConceptNet Query - Consolidated CSV files

3. Hierarchical Data Format (HDF) - ConceptNet PPMI word embeddings 

4. Retrofitted Hierarchical Data Format (HDF) - ConceptNet PPMI word embeddings  after retrofitting


### To-Dos:

1. **[OPEN]** Data collection: 
    * Finalize lists of both gender neutral & stereotypical words
    * For each source, please remember to cite it as reference.
        1. [stjerneskinn - Gender Neutral Words](https://stjerneskinn.com/gender-neutral-words.htm)
        2. [universalenglish.org - Gender Neutral English](https://universalenglish.org/gender-neutral-english/)
        3. [Canadian Mesuem for Human Rights - Gender-Neutral Terms](https://id.humanrights.ca/appendix-b/)
    * Maybe aim for 1000 such words first? Check online repos if there is any lists available --> ALL

2. ConceptNet Scraping:

    a) **[COMPLETED]** https://github.com/commonsense/conceptnet-numberbatch/blob/master/text_to_uri.py —> To convert words found in Point 1 to URI format for API request string
    
    b) **[COMPLETED]** Web-scraping script needed —> ensure we dont bust the ~120 words/minute limit
        
        * OPEN question: Decide what many degrees is needed to capture for edges of each word node, or tree-search methods (BFS, DFS) required to query
    
    c) **[COMPLETED]** To store each raw edge data for a particular node word as JSON file

    d) **[COMPLETED]** To store each specific edge data for a particular node word as CSV file. CSV file with 8 columns:

        * end_id
        * end_label,
        * start_id,
        * start_label,
        * rel_id,
        * surface_text,
        * weight,
        * dataset
    
    e) **[OPEN]** Store in GDrive folder; use [gdown library](https://github.com/wkentaro/gdown) to download to local directory.

3. Research how to convert graphical data into word embeddings
    
    a) NumberBatch: To look into [ConceptNet's Vectors](https://github.com/commonsense/conceptnet5/tree/master/conceptnet5/vectors) library code to convert from graph data to word embeddings --> Kartik
    * Conversion of conceptnet adajacency data into Positive Point-wise Mutual Information (PPMI) word embedding --> Bernard
    
    b) Research on applying retrofitting & expanding retrofiiting to pre-trained word embeddings --> Xue Ling:
    * retrofitting: https://arxiv.org/pdf/1411.4166
    * expanded retrofitting: https://arxiv.org/pdf/1604.01692

4. Fine-tuning existing pre-trained HuggingFace models using word embeddings

    a) TBD

    b) TBD

### References/Sources
* Speer, R., Chin, J. and Havasi, C., 2017, February. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).

* Levy, O., Goldberg, Y. and Dagan, I., 2015. Improving distributional similarity with lessons learned from word embeddings. Transactions of the association for computational linguistics, 3, pp.211-225.


