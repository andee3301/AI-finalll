# Melanoma Classifier

TensorFlow/Keras notebook for classifying melanoma using the Kaggle Melanoma Skin Cancer dataset. Designed to run entirely on Google Colab with GPU acceleration.

## Repository structure
```
melanoma-classifier/
├── notebooks/
│   └── melanoma_training.ipynb
├── src/
│   ├── data_utils.py         # (optional) helpers for local data prep
│   └── model_utils.py        # (optional) model helpers/inference scripts
├── README.md
└── .gitignore
```

## Prerequisites
- Kaggle account with API token (`kaggle.json`). Download from Kaggle > Account > Create API Token.
- Google account for Colab + Google Drive (for saving the trained model).

## Run the notebook on Google Colab
1. Open `notebooks/melanoma_training.ipynb` in the repo and click the "Open in Colab" button (or upload to Colab manually).
2. In Colab: `Runtime > Change runtime type > GPU` (recommended).
3. Run the first cells; when prompted, upload your `kaggle.json`. The notebook will download and unzip the Kaggle dataset automatically.
4. The notebook performs EDA, splits data (train/val/test), builds an EfficientNet model with augmentation and class weights, trains with callbacks, and evaluates on the test split.
5. Connect your Google Drive when prompted; the trained model is saved under `MyDrive/melanoma_models/`.

## Notes
- EfficientNet backbone can be switched between B0 and B3 via the `BACKBONE` flag in the model build cell.
- If the dataset lacks explicit `val`/`test` folders, the notebook creates stratified splits from the training set.
- Adjust `IMG_SIZE`, `BATCH_SIZE`, and `EPOCHS` to balance speed vs. accuracy.

## Local development (optional)
The `src/` folder is provided for future Python helpers (data preprocessing, inference scripts). Keep heavy data artifacts out of version control; `.gitignore` is configured accordingly.
