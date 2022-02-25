# Disaster Response Pipeline Project

## Installation
1. Clone the repo: `git clone https://github.com/atwahsz/Disaster-Response-Pipeline.git`

2. Install the necassary python packages: `pip3 install -r requirements.txt`

## Projet Overview 
This project is focused on detecting emergency events during a disaster (e.g. tsunami, floods, earthquake) using ML techniques.
An app has been designed to ingest raw text (disaster text messages) and classify them into several categories. Then trasmit them into the responsile entities.
Text from various sources has been provided e.g. news, social media, etc. for training the ML models.
The app is based on Flask framework and provides a UI and informative visualizations about the text data received for training.

## Project Structure
- app: Contain files for hosting/running the Web app.
    - __init__.py
    - WordVectorTransformer.py (Not used currently)
    - run.py
    - templates
- data: Contain python code for data preprocessing and database generation.
    - __init__.py
    - disaster_categories.csv
    - disaster_messages.csv
    - DisasterResponse.db
    - ETL Pipeline Preparation.ipynb
    - process_data.py
- models: Contain python code for ML training and model evaluation.
    - __init__.py
    - classifier.pkl
    - ML Pipeline Preparation.ipynb
    - train_classifier.py
    - WordVectorTransformer.py (Not used currently)
- README.md
- requirements.txt

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgments
Many thanks to atwahsz for providing the function for evaluating the ML model's performance!

Repo: https://github.com/atwahsz/Disaster-Response-Pipeline/blob/master/ML%20Pipeline%20Preparation.ipynb