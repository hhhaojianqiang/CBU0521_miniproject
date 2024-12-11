"# CBU0521 Miniproject" 

# CBU0521 Miniproject

This repository contains the submission for the CBU0521 miniproject. The project aims to detect whether narrated stories are true or deceptive using audio features and machine learning techniques.

## Project Structure

- **CBU5201_miniproject_submission.ipynb**: The main Jupyter Notebook containing text, code, and outputs.
- **Python Scripts**: Contains scripts for data processing, visualization, and model training.
  - `ML_miniproject1.py`
  - `data_visualization.py`
  - `model_training.py`
- **CSV File**: Processed feature matrix used for training and testing.
  - `processed_audio_features.csv`
- **Images**: Visualizations generated during the project.
  - `Feature Correlation Matrix.png`
  - `MFCC features distribution.png`
  - `classification report.png`
  - `confusion matrix.png`
   `feature importance.png`

## Instructions to Run the Project

1. Clone this repository:
 ```bash
   git clone https://github.com/hhhaojianqiang/CBU0521_miniproject.git
   cd CBU0521_miniproject

2. Set up the environment:
- Install the required Python libraries listed in the - `requirements.txt file`
- Run the following command
pip install -r requirements.txt

3.Run Jupyter Notebook:

- Launch Jupyter Notebook
	jupyter notebook

- Open `CBU5201_miniproject_submission.ipynb` and execute the cells sequentially

4.Run Python Scripts (Optional):

- You can execute the Python scripts independently
	-`python ML_miniproject1.py`
	-`python data_visualization.py`
	-`python model_training.py`

- **Key Features**
-Dataset Preparation: Processes audio files and attributes from a CSV file to extract meaningful features such as MFCCs, Chroma features, and emotional scores.
-Exploratory Data Analysis: Visualizes class distributions, audio feature histograms, and correlation matrices.
-Model Training: Implements a Random Forest classifier for binary classification.
-Results Analysis: Includes confusion matrices, classification reports, and feature importance plots.