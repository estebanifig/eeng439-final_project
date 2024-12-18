Here is the full README.md file with the dataset download instructions implemented.

# **Chess Piece Classification with EfficientNet**

This project automates the generation, processing, and classification of chessboard images using EfficientNet. It supports generating datasets, sorting images into labeled categories, training a convolutional neural network, and running inference.

---

## **Table of Contents**
1. [Folder Structure](#folder-structure)  
2. [Setup and Installation](#setup-and-installation)  
3. [Data Collection](#data-collection)  
   - Using `datasetCollector.py`  
   - Using `datasetCollector(random).py`  
   - Using `datasetCollector(games).py`  
4. [Data Processing](#data-processing)  
   - Using `chessboardParse.py`  
   - Using `chessboardParse(init).py`  
5. [Data Organization](#data-organization)  
   - Using `dataSorter.py`  
   - Using `dataSorter(init).py`  
6. [Dataset Splitting](#dataset-splitting)  
7. [Training the Model](#training-the-model)  
8. [Evaluating the Model](#evaluating-the-model)  
9. [Running Inference](#running-inference)  
10. [Dataset](#dataset)  

---

## **Folder Structure**

Your project directory is organized as follows:

```plaintext
eeng439-final_project/
│
├── data/                      # All dataset-related files
│   ├── chessboard_screenshots/   # Raw chessboard screenshots and FEN files
│   ├── processed_chessboards/    # Individual chessboard squares
│   ├── dataset/                  # Final labeled dataset
│   └── split_dataset/            # Train, Validation, and Test splits
│
├── models/                    # Trained model weights and class indices
│   ├── final_chess_efficientnet.pth
│   └── class_indices.json
│
├── src/                       # Source code
│   ├── chessboardParse.py       # Splits chessboard into squares (custom FEN)
│   ├── chessboardParse(init).py # Splits chessboard using initial FEN
│   ├── datasetCollector.py      # Collects screenshots from themes
│   ├── datasetCollector(random).py # Collects random chess positions
│   ├── datasetCollector(games).py  # Collects moves from chess games
│   ├── datasetSplit.py          # Splits dataset into train/val/test
│   ├── dataSorter.py            # Sorts images based on FEN
│   ├── dataSorter(init).py      # Sorts images for the initial position
│   ├── dataset_loader.py        # Loads dataset for training/evaluation
│   ├── model.py                 # Defines EfficientNet model
│   ├── train.py                 # Model training script
│   └── evaluate.py              # Model evaluation script
│
├── .gitignore                 # Ignore unnecessary files
├── main.py                    # Main entry point for training and evaluation
└── README.md                  # This file
```

## **Setup and Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/chess-classification.git
cd chess-classification
```
### **2. Install Dependencies**

Create a Python environment and install required libraries:
```bash
pip install -r requirements.txt
```
## **Dataset**

The dataset containing raw chessboard screenshots and corresponding FEN files is available for download.

Download the Dataset
	1.	Use the following Google Drive link (https://drive.google.com/drive/folders/1Iz9qiifqnNKm7FgW6YiZjdmgcHXsspmW?usp=sharing) to download the dataset folder.
	2.	Once downloaded, extract the contents

Where to Place the Dataset
	1.	Place the extracted dataset folder into the data/ directory of this project.
	•	The folder structure should look like this:
```plaintext
eeng439-final_project/
├── data/
│   ├── chessboard_screenshots/   # Place the downloaded dataset here
│   └── (other folders like processed_chessboards, dataset, split_dataset)
```
## **Processing the Dataset**

After placing the dataset in the correct location, follow these steps:
	1.	Split Chessboards into Individual Squares
Use one of the following scripts:
	•	For custom FEN files:
```bash
python src/chessboardParse.py
```

	•	For the initial standard chess position:
```bash
python src/chessboardParse(init).py
```

Output: Individual squares will be saved in data/processed_chessboards.

	2.	Sort Squares into Class Folders
	•	For unique FEN files:
```bash
python src/dataSorter.py
```

	•	For the initial position FEN:
```bash
python src/dataSorter(init).py
```

Output: Labeled images will be saved in data/dataset.

	3.	Split Dataset into Train/Val/Test
Use the datasetSplit.py script:
```bash
python src/datasetSplit.py
```
Default split:
	•	Train: 80%
	•	Validation: 10%
	•	Test: 10%
Output: Split datasets are saved in data/split_dataset.

## **Data Collection (if required)**

### **1. Automating Theme-Based Screenshots**

Use datasetCollector.py to capture screenshots for different board and piece themes.
	•	How to run:
```bash
python src/datasetCollector.py
```


### **2. Generating Random Chess Positions**

Use datasetCollector(random).py to collect chessboards with random FEN positions.
	•	How to run:
```bash
python src/datasetCollector(random).py
```


### **3. Capturing Game Moves**

Use datasetCollector(games).py to collect board states for chess games.
	•	How to run:
```bash
python src/datasetCollector(games).py
```
## **Training the Model**

Run the main.py script to train the EfficientNet model:
```bash
python main.py
```
	•	Outputs:
	•	Model weights: models/final_chess_efficientnet.pth
	•	Class indices: models/class_indices.json

## **Evaluating the Model**

Evaluation runs automatically after training. It checks accuracy on the test set.
	•	How to run:
```bash
python main.py
```
## **Running Inference**

To predict the class of a single chess square image:
	1.	Update paths in src/inference.py:
	•	image_path: Path to the input square image.
	•	model_path: Path to the trained model weights.
	•	class_indices_path: Path to class_indices.json.
	2.	Run the script:
```bash
python src/inference.py
```
Output:

Predicted class: black_pawn

## **Notes**
	•	Ensure your dataset is balanced for better accuracy.
	•	Update screen coordinates when running automation scripts.
	•	Back up your data before processing.

## **License**

This project is licensed under the MIT License.
