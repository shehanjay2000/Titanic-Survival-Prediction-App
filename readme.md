# Titanic Survival Prediction App

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/your-username/titanic-ml-deployment?style=for-the-badge)

A interactive web application built with Streamlit for predicting passenger survival on the Titanic using a Random Forest machine learning model. This project demonstrates a full ML pipeline: data exploration, model training, interactive app development, and cloud deployment.

Live Demo: [Titanic Survival Prediction App](https://your-app-url.streamlit.app/) (Replace with your deployed URL after setup)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This app uses the classic Titanic dataset from Kaggle to train a Random Forest classifier. Users can explore the data, visualize survival patterns, input custom passenger details for predictions, and review model performance metrics. Built as part of a machine learning deployment assignment, it showcases Streamlit for interactive UIs and GitHub/Streamlit Cloud for version control and hosting.

## Features

- **Data Exploration**: Dataset summary, sample view, and interactive filtering by passenger class.
- **Visualizations**: Interactive charts (using Plotly) for survival by class, sex, and age distribution.
- **Prediction**: Real-time survival predictions with probability, using user-input features.
- **Model Performance**: Accuracy, classification report, and confusion matrix visualization.
- **Error Handling**: User-friendly messages for invalid inputs or loading issues.
- **Styling**: Custom CSS for a clean, responsive UI.

## Screenshots

### Data Exploration Page
![Data Exploration](screenshots/data_exploration.png)

### Visualizations Page
![Visualizations](screenshots/visualizations.png)

### Prediction Page
![Prediction](screenshots/prediction.png)

### Model Performance Page
![Model Performance](screenshots/model_performance.png)


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shehanjay2000/Titanic-Survival-Prediction-App.git
   cd Titanic-Survival-Prediction-App
   ```

2. **Set Up Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**:
   - Get `train.csv` from [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data).
   - Rename it to `Titanic-Dataset.csv` and place it in the `data/` folder.

5. **Train the Model** (If not already done):
   - Run the Jupyter notebook: `jupyter notebook notebooks/model_training.ipynb`.
   - This generates `model.pkl`.

## Usage

1. **Run Locally**:
   ```bash
   streamlit run app.py
   ```
   - Access at `http://localhost:8501`.

2. **Navigate the App**:
   - Use the sidebar to switch pages.
   - In "Prediction," enter details and submit for results.
   - Explore visualizations for insights.

## Model Training

The model is trained in `notebooks/model_training.ipynb` using scikit-learn:

- **Preprocessing**: Handle missing values (e.g., Age median imputation), encode categoricals (Sex, Embarked), add features (FamilySize).
- **Models Compared**: Random Forest and Logistic Regression with cross-validation.
- **Best Model**: Random Forest saved as `model.pkl`.

Run the notebook to retrain if needed.

## Deployment

Deployed on Streamlit Cloud for public access.

1. **Prepare**:
   - Ensure all files (including `model.pkl` and dataset) are in GitHub.
   - Use Git LFS for large files if necessary.

2. **Deploy**:
   - Log in to [Streamlit Cloud](https://streamlit.io/cloud).
   - Connect your GitHub repo, select `main` branch, and set `app.py` as the entry point.
   - Deploy and share the URL.

## Project Structure

```
titanic-ml-deployment/
├── app.py                # Streamlit app code
├── requirements.txt      # Dependencies
├── model.pkl             # Trained model
├── data/
│   └── Titanic-Dataset.csv  # Dataset
├── notebooks/
│   └── model_training.ipynb  # Training notebook
├── screenshots/          # App screenshots (optional)
└── README.md             # This file
```

## Troubleshooting

- **Model Not Loading**: Ensure `model.pkl` exists and was trained correctly.
- **Dataset Error**: Verify `data/Titanic-Dataset.csv` path.
- **Prediction Fails**: Check input values; ensure preprocessing matches training.
- **Deployment Issues**: Review Streamlit Cloud logs; update `requirements.txt` if needed.


## Contributing

Contributions welcome! Fork the repo, create a branch, and submit a pull request. Follow best practices: clean code, comments, and tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (create if not present).

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework.
- [Kaggle](https://kaggle.com) for the Titanic dataset.


