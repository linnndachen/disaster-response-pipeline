# Disaster Response Pipeline Project

1. Project Summary

2. Main Files

4. Results

5. Licensing, Authors, and Acknowledgements

6. Tech Stack

7. Instructions

### Main Files: Project Structure

```
├── README.md
├── requirements.txt *** The dependencies we need to install with "pip3 install -r requirements.txt"
├── ap
│   ├── run.py *** See the results via this app
│   ├── templates
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── process_data.py *** ETL pipeline
└── models
    └── train_classifier.py *** Machine-learning model pipline
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
Udacity for providing such a complete Data Science Nanodegree Program
Figure Eight for providing messages dataset to train my model
