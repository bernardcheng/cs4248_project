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

### References/Sources (TO-DO: Tidy up when writing report)

1. [stjerneskinn - Gender Neutral Words](https://stjerneskinn.com/gender-neutral-words.htm)
2. [universalenglish.org - Gender Neutral English](https://universalenglish.org/gender-neutral-english/)