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
- app
- data
- models


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
