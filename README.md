# Autism Classification using Deep Learning
This repository contains code and data for classifying individuals as either having ***Autism or Not*** using deep learning techniques.

# Requirements
The following packages are required to run the code:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Streamlit
# Data
The data used in this project is the Autism Spectrum Disorder Screening Data for Younger Individuals from [Kaggle](https://www.kaggle.com/datasets/cihan063/autism-image-data)

# Code
The code is organized into two main files:

Model.ipynb: 
- This file contains code for preprocessing the raw data and transforming it into a format that can be used for modeling.
- code for training a deep learning model on the preprocessed data and evaluating its performance.

app.py:
- This file Contains code for running the model as a web app.We are using ***Streamlit*** to build a front end for our model. where we can just upload a photo of a person and the model will predict whether the person has autism or not.

# Usage
- Make sure that you have all the required packages installed before running the code.
- Clone the repository to your local machine.
- Open the Model.ipynb file in Jupyter Notebook and run the code.
- On running the Model.ipynb file, the model will be trained and saved as a .h5 file. or you can directly used the model.h5 file which is already trained.
- Make sure that the model.h5 file is in the same directory as the app.py file.
- Run the following command in the terminal to run the web app:

```bash
streamlit run app.py

```


# Results
- The current implementation of the deep learning model achieved an accuracy of 97.36% on the train data and 75% on the test data.
- Since the data was very less to train the model, the model is overfitting on the train data.***But*** This the best results we have got so far on this dataset.

# Limitations
- The data used in this project is limited to a specific population of individuals with autism and may not generalize to other populations.



# Future Work
- Using additional data and features to improve the performance of the model.

- Exploring different deep learning architectures and optimizers to see if they perform better than the current model.

- Conducting further analyses to gain a deeper understanding of the relationships between the features and the target variable.
