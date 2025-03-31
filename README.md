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
* [Speer, R., Chin, J. and Havasi, C., 2017, February. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).](https://arxiv.org/pdf/1612.03975)

* [Levy, O., Goldberg, Y. and Dagan, I., 2015. Improving distributional similarity with lessons learned from word embeddings. Transactions of the association for computational linguistics, 3, pp.211-225.](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00134/43264/Improving-Distributional-Similarity-with-Lessons)

* [Speer, R. and Chin, J., 2016. An ensemble method to produce high-quality word embeddings (2016). arXiv preprint arXiv:1604.01692.](https://arxiv.org/pdf/1604.01692)

* [Speer, R. and Lowry-Duda, J., 2017. Conceptnet at semeval-2017 task 2: Extending word embeddings with multilingual relational knowledge. arXiv preprint arXiv:1704.03560.](https://arxiv.org/pdf/1704.03560)

* [Faruqui, M., Dodge, J., Jauhar, S.K., Dyer, C., Hovy, E. and Smith, N.A., 2014. Retrofitting word vectors to semantic lexicons. arXiv preprint arXiv:1411.4166.](https://arxiv.org/pdf/1411.4166)

* [Jentzsch, S., Schramowski, P., Rothkopf, C. and Kersting, K., 2019, January. Semantics derived automatically from language corpora contain human-like moral choices. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (pp. 37-44).](https://www.aiml.informatik.tu-darmstadt.de/papers/jentzsch2019aies_moralChoiceMachine.pdf)

* [Bolukbasi, T., Chang, K.W., Zou, J.Y., Saligrama, V. and Kalai, A.T., 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural information processing systems, 29.](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)