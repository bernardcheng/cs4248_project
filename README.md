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

### To-Do:

1. Determine Lists of both gender neutral / stereotypical words —> Maybe aim for 1000 such words first? Check online repos if there is any lists available --> ALL

2. ConceptNet Scraping: --> Bernard

    a) https://github.com/commonsense/conceptnet-numberbatch/blob/master/text_to_uri.py —> To convert words found in Point 1 to URI format for API request string
    
    b) Web-scraping script needed —> ensure we dont bust the ~120 words/minute limit + decide what many degrees is needed to capture for edges of each word node
    
    c) What to scrape: To store each edge data for a particular node word as JSON/YAML file
    
    --> Store in GDrive folder; use [gdown library](https://github.com/wkentaro/gdown) to download to local directory. 

    --> File Formats:
     * Raw JSON
     * CSV file with 4 columns:
        * input word:
        * edge name: 
        * weight:
        * rel:

3. Research how to convert graphical data into word embeddings --> Xue Ling &  Kartik
    
    a) NumberBatch: To look into https://github.com/commonsense/conceptnet5/tree/master/conceptnet5/vectors code base to convert from graph data to  word embeddings
    
    b) Look into applying expanded retrofitting to pre-trained word embeddings

        * Speer, R., Chin, J. and Havasi, C., 2017, February. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).

        * Levy, O., Goldberg, Y. and Dagan, I., 2015. Improving distributional similarity with lessons learned from word embeddings. Transactions of the association for computational linguistics, 3, pp.211-225.

### References/Sources (TO-DO: Tidy up when writing report)

1. [stjerneskinn - Gender Neutral Words](https://stjerneskinn.com/gender-neutral-words.htm)
2. [universalenglish.org - Gender Neutral English](https://universalenglish.org/gender-neutral-english/)
3. [Canadian Mesuem for Human Rights - Gender-Neutral Terms](https://id.humanrights.ca/appendix-b/)