# Disaster Response Pipeline Project

### Project Summary

This project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset containing real messages that were sent during disaster events. The goal is to create a NLP pipeline to categorize these messages so that they can be sent to an appropriate disaster relief agency.

The outcome of the project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


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

### Results:


### Tech Stack:
Tech stack will include:

- **SQLAlchemy ORM** to be ORM library of choice
- **Sqlite 3** as database of choice
- **Python3** and **Flask** as our server language and server framework
- **HTML, CSS,** and **Javascript** with Bootstrap 3 for our website's frontend

### Instructions:
1. Download this repository as a ZIP file by clicking the top right green button.

2. Upzip the file and run `pip3 install -r requirements.txt` to install the dependencies.

3. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/

### Licensing, Authors, and Acknowledgements

[Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program.

[Figure  Eight](https://appen.com/) for providing messages dataset to train my model.
