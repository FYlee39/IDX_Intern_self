# Housing Price Prediction Web App

This repository contains an end-to-end housing price prediction project built from California residential real estate data. It includes exploratory notebooks, feature engineering, model experiments, and a Streamlit app that serves the final trained model for batch CSV prediction.

The project is intended to show a practical machine learning workflow: cleaning raw real estate records, engineering model-ready features, comparing regression approaches, serializing a trained pipeline, and building a small reviewable inference interface.

## What the App Does

- Accepts a CSV file of housing records.
- Validates that the uploaded file contains the model's required feature columns.
- Loads the trained model from `web_app/model.pkl`.
- Generates predicted log close prices for each uploaded row.
- Lets users download prediction results as a CSV file.
- Provides a model information page for inspecting the serialized pipeline.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- func.py                         # Data loading, cleaning, feature engineering, and modeling helpers
|-- plot_func.py                    # Plotting helpers used during analysis
|-- models.py                       # Early model experiment imports / scratch module
|-- *_model*.ipynb                  # Model development notebooks
|-- cleaning.ipynb                  # Initial data cleaning workflow
|-- eda.ipynb                       # Exploratory data analysis
|-- feature_engineering.ipynb       # Feature creation workflow
|-- web_app/
|   |-- main.py                     # Streamlit multipage entry point
|   |-- home.py                     # App overview and required schema
|   |-- upload_predict.py           # CSV upload, validation, and prediction page
|   |-- model_info.py               # Model inspection page
|   |-- about.py                    # Project context and limitations
|   |-- state_helpers.py            # Streamlit session-state helpers
|   |-- utils.py                    # Model loading, validation, and prediction helpers
|   `-- model.pkl                   # Serialized trained model pipeline
`-- tests/
    `-- test_web_app_utils.py       # Unit tests for validation and prediction helpers
```

Large raw and processed CSV files are intentionally ignored by Git. Keep private data files outside version control and document how to recreate them.

## Technologies Used

- Python
- Streamlit
- pandas and NumPy
- scikit-learn-compatible modeling pipelines
- XGBoost and LightGBM experiments
- category_encoders for categorical preprocessing
- joblib for model serialization
- Jupyter notebooks for EDA and model development

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit App

```powershell
streamlit run web_app/main.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Input Data Requirements

Uploaded CSV files must include every feature expected by the trained pipeline. The full required schema is displayed on the app home page and is defined in `web_app/utils.py` as `REQUIRED_FEATURES`.

A partial example of the expected style:

```csv
LivingArea,BedroomsTotal,BathroomsTotalInteger,Latitude,Longitude,YearBuilt,City,CountyOrParish
1800,3,2,34.05,-118.24,1998,Los Angeles,Los Angeles
```

The example above is not a complete upload file. It shows representative columns only; the model requires the full feature set.

## Run Tests

The current tests cover reusable app utility behavior without launching Streamlit:

```powershell
python -m unittest discover -s tests
```

You can also run a syntax check:

```powershell
python -m compileall web_app func.py plot_func.py models.py
```

## Technical Decisions

- The app uses a serialized model pipeline so inference stays consistent with the training-time preprocessing assumptions.
- Required features are centralized in `web_app/utils.py` to keep validation, feature ordering, and UI documentation aligned.
- Streamlit session state stores uploaded data, validation state, and prediction results across pages.
- Data access credentials are read from environment variables instead of being committed as source-code defaults.

## Limitations

- Predictions are estimates and should not be used as official property valuations.
- The uploaded CSV must match the training schema; the app does not currently rebuild engineered features from raw MLS exports.
- Several notebooks are exploratory and could be further consolidated into a reproducible training pipeline.
