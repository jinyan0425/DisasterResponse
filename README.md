# Classifying Disaster Responses

![image](https://user-images.githubusercontent.com/90875339/197631742-e89cbd5b-ba85-4523-a533-4bedc97f0475.png)

## Table of Contents
1. [Project Overview](#ProjectOverview)
   1. [Data Description](#DataDescription)
   2. [Repo Description](#RepoDescription)
3. [Getting Started](#GettingStarted)
4. [Classification Model](#ClassificationModel)
5. [Lisense, Author & Acknowledgements](#ALA)

<a name="ProjectOverview"></a>
## Project Overview
Following a disaster, there are millions of communications, and classifying these communications (e.g., social media posts including tweets and text messages) is crucial for disaster response organizations to form teams effectively and for disaster response professionals to assist disaster victims timely. This project develops a logistic-regression-based classifier to classify pre-labelled post-disaster tweets (N = 30k+), which would contribute to the intensive work on disaster response classification. 

<a name="DataDescription"></a>
### Data Description
![Data](https://user-images.githubusercontent.com/90875339/197644073-e9c06d09-b781-4e7d-823e-cce5e24aa363.png)

<a name="RepoDescription"></a>
### Repo Description
This repo consists of three folders:  
* ```app```, which contains the script (i.e.,[run.py](app/run.py)) for running the Flask Web App. This [interactive app] (http://0.0.0.0:3000/) can classify any message users input.
![example](https://user-images.githubusercontent.com/90875339/197644905-ec2ee4d9-f139-4446-aaca-4f5ac3784fa5.png)
* ```data```, which contains  the script(i.e.[process_data.py](data/process_data.py))for data collection and wrangling, as well as the data (i.e., pre-labeled tweets) for developing and testing the model
* ```models```, which contains the script(i.e., [train_classifier.py](models/train_classifier.py))for processing the natural languages and developing the Machine Learning (Classifier) pipeline

<a name="GettingStarted"></a>
## Getting Started

<a name="Dependencies"></a>
### Dependencies
* Python 3.5+
* SQLlite Database Libraqries: ```SQLalchemy```
* Data Wrangling Libraries: ```Pandas```
* Natural Language Process Libraries: ```NLTK```
* Machine Learning Libraries: ```NumPy```, ```Sciki-Learn```
* Web App and Data Visualization: ```Flask```, ```Plotly```
* Others: ```Pickle```

<a name="Installation"></a>
### Installing
To clone the git repository:
```
git clone https://github.com/jinyan0425/DisasterResponse.git
```
<a name="ClassificationModel"></a>
## Classification Model
* The final classifier used was Multi-label Logistic Regression Classifier, and the estimator and paramaters information can be found at [classifier.pkl](models/classifier.pkl).
* The F1 score of the classifier is decent (weighted: 0.67 and micro: 0.64, because the classes for most message labels (categories) are unbalanced). The model may be improved with more data and more features (beyound the scope of the current project).
* For the rationales behind the model development and evaluation, please refer to the [notebook](https://github.com/jinyan0425/notebooks/blob/02de59fe0c7b10f061fb34940c449f64f32b5163/DisasterResponse_Prep/train_classifier_prep.ipynb).

<a name="ALA"></a>
## Author, License & Acknowledgements

### Author
Jinyan Xiang, find me on [LinkedIn](https://www.linkedin.com/in/jinyanxiang/)

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Acknowledgements
* [Udacity](https://www.udacity.com/) as the project creater
* [Figure Eight](https://www.figure-eight.com/) as the data provider

