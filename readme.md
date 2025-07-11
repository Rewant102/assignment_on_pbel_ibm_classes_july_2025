# 💼 Salary Prediction App

An intelligent Streamlit app that predicts employee salaries using either:
- ✅ A locally trained Machine Learning model (XGBoost)
- 🌐 IBM Watsonx deployed model via API

This project showcases real-world MLOps with hybrid prediction options, built for smart HR analytics and salary forecasting.

---
### Project Structure
📁 my-smart-predictor/
│
├── app.py # Streamlit app
├── requirements.txt # Python dependencies
├── runtime.txt # Optional: Python version pinning
├── .gitignore # Clean Git tracking
├── .env # Contains API key (excluded from GitHub)
│
├── salary_model_clean.pkl # Local XGBoost model
├── scaler_clean.pkl # Feature scaler
├── label_encoders_clean.pkl # Encoders for department/position
│
└── .streamlit/
└── config.toml # Cloud config file

## 🚀 Live App

🔗 [Click to launch the app](https://rewant102-assignment-on-pbel-ibm-classes-july-2025.streamlit.app)

> Predict salaries instantly using either your own model or IBM’s cloud deployment!

---

## 🧠 Features

✅ Dual Prediction Mode:
- `Local Model`: Runs with `.pkl` files stored in the project
- `Watsonx API`: Sends input data to IBM Cloud and returns predicted salary

🎯 User Inputs:
- Age
- Country
- Department
- Position
- Years of Experience

📊 Output:
- Predicted Salary displayed in ₹ (Indian Rupees)

---

## 📦 Technologies Used

| Tech           | Usage                            |
|----------------|----------------------------------|
| `Streamlit`    | UI for app                       |
| `scikit-learn` | Preprocessing + Label Encoding   |
| `XGBoost`      | Local ML model                   |
| `joblib`       | Load model + encoders            |
| `IBM Watsonx`  | Cloud prediction endpoint        |
| `python-dotenv`| Local `.env` support             |

---

## 🛠️ Project Structure



---

## 🧪 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Rewant102/assignment_on_pbel_ibm_classes_july_2025.git
cd assignment_on_pbel_ibm_classes_july_2025

# 2. (Optional) Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Watson API key in a .env file
echo WATSON_API_KEY=your_actual_key_here > .env

# 5. Run the app
streamlit run app.py

🌐 Deploy to Streamlit Cloud
Push your app to GitHub

Visit streamlit.io/cloud

Connect your GitHub repo and select:

main branch

app.py as entry point

Go to "Advanced Settings" → Secrets and paste:

Copy and edit 
WATSON_API_KEY = "your_actual_key_here"
✅ Done — your app will deploy in seconds!

✨ Credits
Built by Rewant Prajapati
Project under: PBEL IBM AI Cloud Training – July 2025

📬 Contact
📧 rewantprajapati102@gmail.com
🔗 LinkedIn  "www.linkedin.com/in/rewant-prajapati-a968592b7"
💻 GitHub     "https://github.com/Rewant102"

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<this is the part form watson.ai studio >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Pipeline 8 Notebook - AutoAI Notebook v2.1.7

Consider these tips for working with an auto-generated notebook:
- Notebook code generated using AutoAI will execute successfully. If you modify the notebook, we cannot guarantee it will run successfully.
- This pipeline is optimized for the original data set. The pipeline might fail or produce sub-optimal results if used with different data.  If you want to use a different data set, consider retraining the AutoAI experiment to generate a new pipeline. For more information, see <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/autoai-notebook.html">Cloud Platform</a>. 
- Before modifying the pipeline or trying to re-fit the pipeline, consider that the code converts dataframes to numpy arrays before fitting the pipeline (a current restriction of the preprocessor pipeline).

<a id="content"></a>
## Notebook content

This notebook contains a Scikit-learn representation of AutoAI pipeline. This notebook introduces commands for retrieving data, training the model, and testing the model. 

Some familiarity with Python is helpful. This notebook uses Python 3.11 and scikit-learn 1.3.
## Notebook goals

-  Scikit-learn pipeline definition
-  Pipeline training 
-  Pipeline evaluation

## Contents

This notebook contains the following parts:

**[Setup](#Setup)**<br>
&nbsp;&nbsp;[Package installation](#Package-installation)<br>
&nbsp;&nbsp;[AutoAI experiment metadata](#AutoAI-experiment-metadata)<br>
&nbsp;&nbsp;[watsonx.ai connection](#watsonx.ai-connection)<br>
**[Pipeline inspection](#Pipeline-inspection)** <br>
&nbsp;&nbsp;[Read training data](#Read-training-data)<br>
&nbsp;&nbsp;[Create pipeline](#Create-pipeline)<br>
&nbsp;&nbsp;[Train pipeline model](#Train-pipeline-model)<br>
&nbsp;&nbsp;[Test pipeline model](#Test-pipeline-model)<br>
**[Store the model](#Store-the-model)**<br>
**[Summary and next steps](#Summary-and-next-steps)**<br>
**[Copyrights](#Copyrights)**
<a id="setup"></a>
# Setup
<a id="install"></a>
## Package installation
Before you use the sample code in this notebook, install the following packages:
 - ibm-watsonx-ai,
 - autoai-libs,
 - scikit-learn,
 - xgboost

!pip install ibm-watsonx-ai | tail -n 1
!pip install autoai-libs~=2.0 | tail -n 1
!pip install scikit-learn==1.3.* | tail -n 1
!pip install -U lale~=0.8.3 | tail -n 1
!pip install xgboost==2.0.* | tail -n 1
Filter warnings for this notebook.
import warnings

warnings.filterwarnings('ignore')
<a id="variables_definition"></a>
## AutoAI experiment metadata
The following cell contains the training data connection details.  
**Note**: The connection might contain authorization credentials, so be careful when sharing the notebook.
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers import ContainerLocation

training_data_references = [
    DataConnection(
        data_asset_id='cb940aef-775c-4aac-be3c-ec07e06d6035'
    ),
]
training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/1fb42989-a21b-4a2c-9469-fd62525eb3a9/wml_data/fec0a58d-f068-4e18-9935-a0ee7d9afaa2/data/automl',
        model_location='auto_ml/1fb42989-a21b-4a2c-9469-fd62525eb3a9/wml_data/fec0a58d-f068-4e18-9935-a0ee7d9afaa2/data/automl/model.zip',
        training_status='auto_ml/1fb42989-a21b-4a2c-9469-fd62525eb3a9/wml_data/fec0a58d-f068-4e18-9935-a0ee7d9afaa2/training-status.json'
    )
)
The following cell contains input parameters provided to run the AutoAI experiment in Watson Studio.
experiment_metadata = dict(
    prediction_type='regression',
    prediction_column='Salary',
    holdout_size=0.1,
    scoring='neg_root_mean_squared_error',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference,
    deployment_url='https://au-syd.ml.cloud.ibm.com',
    project_id='38a16326-b848-49d5-bde2-06591a1970b9',
    drop_duplicates=True,
    include_batched_ensemble_estimators=[],
    feature_selector_mode='auto'
)
## Set `n_jobs` parameter to the number of available CPUs
import os, ast
CPU_NUMBER = 4
if 'RUNTIME_HARDWARE_SPEC' in os.environ:
    CPU_NUMBER = int(ast.literal_eval(os.environ['RUNTIME_HARDWARE_SPEC'])['num_cpu'])
<a id="connection"></a>
## watsonx.ai connection

This cell defines the credentials required to work with the watsonx.ai Runtime.

**Action**: Provide the IBM Cloud apikey, For details, see [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey).
import getpass

api_key = getpass.getpass("Please enter your api key (press enter): ")
from ibm_watsonx_ai import Credentials

credentials = Credentials(
    api_key=api_key,
    url=experiment_metadata['deployment_url']
)
from ibm_watsonx_ai import APIClient

client = APIClient(credentials)

if 'space_id' in experiment_metadata:
    client.set.default_space(experiment_metadata['space_id'])
else:
    client.set.default_project(experiment_metadata['project_id'])

training_data_references[0].set_client(client)
<a id="inspection"></a>
# Pipeline inspection
<a id="read"></a>
## Read training data

Retrieve training dataset from AutoAI experiment as pandas DataFrame.

**Note**: If reading data results in an error, provide data as Pandas DataFrame object, for example, reading .CSV file with `pandas.read_csv()`.

It may be necessary to use methods for initial data pre-processing like: e.g. `DataFrame.dropna()`, `DataFrame.drop_duplicates()`, `DataFrame.sample()`, and outliers handling:

```
from autoai_libs.utils.outliers_mitigation import remove_outliers

df = remove_outliers(df, columns=[experiment_metadata['prediction_column']])
```

X_train, X_test, y_train, y_test = training_data_references[0].read(experiment_metadata=experiment_metadata, with_holdout_split=True, use_flight=True)
<a id="preview_model_to_python_code"></a>
## Create pipeline
In the next cell, you can find the Scikit-learn definition of the selected AutoAI pipeline.
#### Import statements.
from autoai_libs.transformers.exportable import ColumnSelector
from autoai_libs.transformers.date_time.date_time_transformer import (
    DateTransformer,
)
from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
from autoai_libs.transformers.exportable import boolean2float
from autoai_libs.transformers.exportable import CatImputer
from autoai_libs.transformers.exportable import CatEncoder
import numpy as np
from autoai_libs.transformers.exportable import float32_transform
from sklearn.pipeline import make_pipeline
from autoai_libs.transformers.exportable import FloatStr2Float
from autoai_libs.transformers.exportable import NumImputer
from autoai_libs.transformers.exportable import OptStandardScaler
from sklearn.pipeline import make_union
from autoai_libs.transformers.exportable import NumpyPermuteArray
from autoai_libs.cognito.transforms.transform_utils import TAM
from sklearn.decomposition import PCA
from autoai_libs.cognito.transforms.transform_utils import FS1
from autoai_libs.cognito.transforms.transform_utils import TA1
import autoai_libs.utils.fc_methods
from xgboost import XGBRegressor
#### Pre-processing & Estimator.
column_selector_0 = ColumnSelector(columns_indices_list=[1, 2, 3, 4, 5, 6])
date_transformer_0 = DateTransformer(
    column_headers_list=[
        "Employee_Name", "Age", "Country", "Department", "Position",
        "Joining_Date",
    ],
    date_column_indices=[5],
    missing_values_reference_list=["?", "", "-", float("nan")],
    options=["all"],
)
numpy_column_selector_0 = NumpyColumnSelector(
    columns=[0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
)
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=[
        "char_str", "float_int_num", "char_str", "char_str", "char_str",
        "int_num", "int_num", "int_num", "int_num", "int_num",
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
    misslist_list=[[], [], [], [], [], [], [], [], [], []],
)
numpy_replace_missing_values_0 = NumpyReplaceMissingValues(
    filling_values=float("nan"), missing_values=[]
)
numpy_replace_unknown_values = NumpyReplaceUnknownValues(
    filling_values=float("nan"),
    filling_values_list=[
        float("nan"), 100001, float("nan"), float("nan"), float("nan"),
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
)
cat_imputer = CatImputer(
    missing_values=float("nan"),
    sklearn_version_family="1",
    strategy="most_frequent",
)
cat_encoder = CatEncoder(
    dtype=np.float64,
    handle_unknown="error",
    sklearn_version_family="1",
    encoding="ordinal",
    categories="auto",
)
pipeline_0 = make_pipeline(
    column_selector_0,
    date_transformer_0,
    numpy_column_selector_0,
    compress_strings,
    numpy_replace_missing_values_0,
    numpy_replace_unknown_values,
    boolean2float(),
    cat_imputer,
    cat_encoder,
    float32_transform(),
)
column_selector_1 = ColumnSelector(columns_indices_list=[1, 2, 3, 4, 5, 6])
date_transformer_1 = DateTransformer(
    column_headers_list=[
        "Employee_Name", "Age", "Country", "Department", "Position",
        "Joining_Date",
    ],
    date_column_indices=[5],
    missing_values_reference_list=["?", "", "-", float("nan")],
    options=["all"],
)
numpy_column_selector_1 = NumpyColumnSelector(columns=[5, 9])
float_str2_float = FloatStr2Float(
    dtypes_list=["float_int_num", "int_num"], missing_values_reference_list=[]
)
numpy_replace_missing_values_1 = NumpyReplaceMissingValues(
    filling_values=float("nan"), missing_values=[]
)
num_imputer = NumImputer(missing_values=float("nan"), strategy="median")
opt_standard_scaler = OptStandardScaler(use_scaler_flag=False)
pipeline_1 = make_pipeline(
    column_selector_1,
    date_transformer_1,
    numpy_column_selector_1,
    float_str2_float,
    numpy_replace_missing_values_1,
    num_imputer,
    opt_standard_scaler,
    float32_transform(),
)
union = make_union(pipeline_0, pipeline_1)
numpy_permute_array = NumpyPermuteArray(
    axis=0, permutation_indices=[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 5, 9]
)
tam = TAM(
    tans_class=PCA(),
    name="pca",
    col_names=[
        "Employee_Name", "Age", "Country", "Department", "Position",
        "NewDateFeature_0_FloatTimestamp(Joining_Date)",
        "NewDateFeature_1_Year(Joining_Date)",
        "NewDateFeature_2_Month(Joining_Date)",
        "NewDateFeature_3_Week(Joining_Date)",
        "NewDateFeature_4_DayOfYear(Joining_Date)",
        "NewDateFeature_5_DayOfMonth(Joining_Date)",
        "NewDateFeature_6_DayOfWeek(Joining_Date)",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
fs1_0 = FS1(
    cols_ids_must_keep=range(0, 12),
    additional_col_count_to_keep=12,
    ptype="regression",
)
ta1 = TA1(
    fun=np.square,
    name="square",
    datatypes=["numeric"],
    feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical],
    col_names=[
        "Employee_Name", "Age", "Country", "Department", "Position",
        "NewDateFeature_0_FloatTimestamp(Joining_Date)",
        "NewDateFeature_1_Year(Joining_Date)",
        "NewDateFeature_2_Month(Joining_Date)",
        "NewDateFeature_3_Week(Joining_Date)",
        "NewDateFeature_4_DayOfYear(Joining_Date)",
        "NewDateFeature_5_DayOfMonth(Joining_Date)",
        "NewDateFeature_6_DayOfWeek(Joining_Date)", "pca_0", "pca_1", "pca_2",
        "pca_3", "pca_4", "pca_5", "pca_6", "pca_7", "pca_8", "pca_9",
        "pca_10", "pca_11",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
fs1_1 = FS1(
    cols_ids_must_keep=range(0, 12),
    additional_col_count_to_keep=12,
    ptype="regression",
)
xgb_regressor = XGBRegressor(
    gamma=0,
    learning_rate=0.08613703421236957,
    max_depth=1,
    min_child_weight=2,
    missing=float("nan"),
    n_estimators=50,
    n_jobs=CPU_NUMBER,
    random_state=33,
    reg_alpha=0,
    reg_lambda=0.12639088967076506,
    subsample=0.9964644386230973,
    verbosity=0,
    silent=False,
)

#### Pipeline.
pipeline = make_pipeline(
    union, numpy_permute_array, tam, fs1_0, ta1, fs1_1, xgb_regressor
)
<a id="train"></a>
## Train pipeline model

### Define scorer from the optimization metric
This cell constructs the cell scorer based on the experiment metadata.
from sklearn.metrics import make_scorer

from autoai_libs.scorers.scorers import neg_root_mean_squared_error

scorer = make_scorer(neg_root_mean_squared_error)
<a id="test_model"></a>
### Fit pipeline model
In this cell, the pipeline is fitted.
pipeline.fit(X_train.values, y_train.values.ravel());
<a id="test_model"></a>
## Test pipeline model
Score the fitted pipeline with the generated scorer using the holdout dataset.
score = scorer(pipeline, X_test.values, y_test.values)
print(score)
pipeline.predict(X_test.values[:5])
<a id="saving"></a>
## Store the model

In this section you will learn how to store the trained model.
model_metadata = {
    client.repository.ModelMetaNames.NAME: 'P8 - Pretrained AutoAI pipeline'
}

stored_model_details = client.repository.store_model(model=pipeline, meta_props=model_metadata, experiment_metadata=experiment_metadata)
Inspect the stored model details.
stored_model_details
<a id="deployment"></a>
## Create online deployment
You can use the commands below to promote the model to space and create online deployment (web service).

<a id="working_spaces"></a>
### Working with spaces

In this section you will specify a deployment space for organizing the assets for deploying and scoring the model. If you do not have an existing space, you can use <a href="https://au-syd.dai.cloud.ibm.com/ml-runtime/dashboard?context=wx">Deployment Spaces Dashboard</a> to create a new space, following these steps:

- Click **New Deployment Space**.
- Create an empty space.
- Select Cloud Object Storage.
- Select watsonx.ai Runtime and press **Create**.
- Copy `space_id` and paste it below.

**Tip**: You can also use the API to prepare the space for your work. Learn more [here](https://github.com/IBM/watson-machine-learning-samples/blob/master/cloud/notebooks/python_sdk/instance-management/Space%20management.ipynb).

**Info**: Below cells are `raw` type - in order to run them, change their type to `code` and run them (no need to restart the notebook). You may need to add some additional info (see the **action** below).

**Action**: Assign or update space ID below.

space_id = "PUT_YOUR_SPACE_ID_HERE"

model_id = client.spaces.promote(asset_id=stored_model_details["metadata"]["id"], source_project_id=experiment_metadata["project_id"], target_space_id=space_id)
#### Prepare online deployment
client.set.default_space(space_id)

deploy_meta = {
        client.deployments.ConfigurationMetaNames.NAME: "Incrementally trained AutoAI pipeline",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }

deployment_details = client.deployments.create(artifact_uid=model_id, meta_props=deploy_meta)
deployment_id = client.deployments.get_id(deployment_details)
#### Test online deployment
import pandas as pd

scoring_payload = {
    "input_data": [{
        'values': pd.DataFrame(X_test[:5])
    }]
}

client.deployments.score(deployment_id, scoring_payload)
<a id="cleanup"></a>
### Deleting deployment
You can delete the existing deployment by calling the `client.deployments.delete(deployment_id)` command.
To list the existing web services, use `client.deployments.list()`.
<a id="summary_and_next_steps"></a>
# Summary and next steps
You successfully completed this notebook!
You learned how to use AutoAI pipeline definition to train the model.
Check out our [Online Documentation](https://www.ibm.com/cloud/watson-studio/autoai) for more samples, tutorials, documentation, how-tos, and blog posts.
<a id="copyrights"></a>
### Copyrights

Licensed Materials - Copyright © 2025 IBM. This notebook and its source code are released under the terms of the ILAN License. Use, duplication disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

**Note:** The auto-generated notebooks are subject to the International License Agreement for Non-Warranted Programs (or equivalent) and License Information document for Watson Studio Auto-generated Notebook (License Terms), such agreements located in the link below. Specifically, the Source Components and Sample Materials clause included in the License Information document for Watson Studio Auto-generated Notebook applies to the auto-generated notebooks.  

By downloading, copying, accessing, or otherwise using the materials, you agree to the <a href="https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BYC7LF">License Terms</a>






<<<<<<<<<<<<<<<<<<<<<<<<<<  self built model which is integrated  with the AutoAI pipeline  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Salary Prediction System with XGBoost

## Project Overview

This project focuses on building a machine learning model to predict employee salaries based on a structured dataset of employee records. The model incorporates both categorical and numerical variables and is engineered to process real-world data, accounting for inconsistencies in job titles and departments. The overarching goal is to create a scalable, robust solution for human resources analytics and compensation planning.

---

## Dataset Details

- **Source File:** `employee_records.csv`
- **Record Count:** 30,000 employee entries
- **Attributes:** 8 columns including employee identifiers, job details, salaries, and joining dates

This dataset reflects a broad cross-section of industries, job functions, and geographical locations, offering a diverse foundation for training a predictive model that generalizes well across scenarios.

---

## Preprocessing Pipeline

1. **Irrelevant Column Removal**

   - Excluded `Employee_ID` and `Employee_Name` as they offer no predictive value
   - Removed personal identifiers to prevent overfitting and support data privacy

2. **Experience Feature Engineering**

   - Derived `YearsExperience` from the `Joining_Date` by calculating the number of years from the current date
   - Applied robust datetime parsing to handle inconsistencies or formatting issues

3. **Categorical Standardization**

   - Consolidated inconsistent entries in `Position` and `Department` using controlled vocabularies
     - e.g., "Software Engineer", "Engineer" → "Developer"
     - e.g., "Tech", "IT" → "Engineering"
   - These mappings reduced data fragmentation and improved model stability

4. **Missing Value Handling**

   - Eliminated records with null or incomplete data in critical fields
   - Conducted post-cleaning validation to ensure dataset quality and consistency

5. **Feature Encoding and Scaling**

   - Encoded categorical fields (`Country`, `Department`, `Position`) using `LabelEncoder`
   - Applied `StandardScaler` to numeric variables to support optimal model training

---

## Model Training

- **Algorithm:** `XGBRegressor` (an efficient and scalable gradient boosting algorithm)
- **Key Hyperparameters:**
  - `n_estimators`: 200
  - `learning_rate`: 0.05
  - `max_depth`: 6
- **Data Split:** 80% training / 20% validation
- **Training Strategy:** Designed to capture complex, non-linear patterns and interactions

### Model Performance

- **R² Score:** \~0.82 (indicates high predictive accuracy)
- **RMSE:** \~₹17,000 (reasonable average error across job types)
- These metrics confirm that the model is well-calibrated and generalizable

---

## Output Artifacts

- `salary_model_clean.pkl`: The trained and serialized XGBoost regression model
- `scaler_clean.pkl`: The scaler object for numeric features
- `label_encoders_clean.pkl`: Serialized encoders for transforming categorical fields

---

## Salary Prediction Pipeline (`jobl_clean.py`)

- Accepts new input with possibly inconsistent category labels
- Automatically maps raw values (e.g., job titles, departments) to their normalized equivalents
- Applies consistent encoding and scaling to match the trained model's expectations
- Predicts salary using the pre-trained XGBoost model

### Example Input

```json
{
  "Age": 30,
  "Country": "India",
  "Department": "IT",
  "Position": "Software Engineer",
  "YearsExperience": 5
}
```

### Example Output

```
Predicted Salary: ₹30,000 - ₹50,000 (depending on input features)
```

---

## Future Enhancements

- Add support for batch predictions from uploaded CSV files
- Develop a user-friendly dashboard using Streamlit for real-time HR usage
- Deploy the model as an API for integration with enterprise HR systems
- Implement monitoring, versioning, and logging for production-level deployment

---

## Conclusion

This project successfully delivered a robust and scalable salary prediction solution powered by machine learning. The system effectively handles real-world data challenges like inconsistent labels, missing values, and scaling requirements. Its modular design makes it suitable for deployment as a standalone tool or as part of a larger HR analytics platform.

