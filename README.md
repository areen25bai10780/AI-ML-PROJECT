#Fake News Detection System
A Machine Learning-based web application that detects whether a news article or headline is **FAKE** or **REAL** using Natural Language Processing (NLP) techniques and multiple classification models.
 Features
 Detects fake vs real news
 Uses Machine Learning models:
* Logistic Regression (Primary)
* Decision Tree (Comparison)
*  Displays prediction confidence scores
*  Generates accuracy comparison graph
*  Flask-based web interface
*  REST API for training and prediction
*  Auto dataset generation (no external download needed)

## How It Works
1. Dataset is generated (fake + real news)
2. Text data is cleaned and preprocessed
3. TF-IDF converts text into numerical form
4. Models are trained on dataset
5. User inputs news text
6. System predicts:
 * Fake ❌
 * Real ✅

 Tech Stack

* **Language:** Python
* **Framework:** Flask
* **Libraries:**
*  * pandas
  * numpy
  * scikit-learn
  * matplotlib

#  Project Structure

textproject/
│
├── app.py                 # Flask web server
├── model.py               # ML logic (dataset, training, prediction)
├── requirements.txt       # Dependencies
│
├── data/
│   └── news.csv           # Generated dataset
│
├── models/
│   ├── vectorizer.pkl
│   ├── logistic_regression.pkl
│   └── decision_tree.pkl
│
├── static/
│   └── plots/
│       └── accuracy.png   # Accuracy graph
│
└── templates/
    └── index.html         # Frontend UI




##Installation & Setup

###  Clone Repository


git clone https://github.com/your-username/your-repo-name
cd your-repo-name


###  Install Dependencies


pip install -r requirements.txt


### 3️ Run Application


python app.py



### 4️ Open in Browser


http://127.0.0.1:5000/




##  Usage

### 🔹 Step 1: Train Model

* Click **"Train Models"** button (if UI present)
* OR call API: `/api/train`



###  Step 2: Predict News

Enter news text and get:

* Prediction (FAKE / REAL)
* Confidence score


##  API Endpoints

###  Check Status

```
GET /api/status
```



###  Train Models


pOST /api/train



###  Predict News


POST /api/predict


**Request Body:**


{
  "text": "Breaking news example"
}


**Response:**
{
  "status": "success",
  "results": {
    "logistic_regression": {
      "prediction": "FAKE",
      "confidence": 98.5
    },
    "decision_tree": {
      "prediction": "FAKE",
      "confidence": 92.3
    },
    "final_verdict": "FAKE"
  }
}

## Model Details

| Model               | Role             |
| ------------------- | ---------------- |
| Logistic Regression | Final Prediction |
| Decision Tree       | Comparison       |



##  Output

* Fake News 
* Real News 
* Confidence Score (%)
* Accuracy Graph (saved in `/static/plots/`)


##  Limitations

* Uses synthetic dataset
* Not trained on real-world large data
* Limited vocabulary

## Future Improvements

* Use real datasets (Kaggle)
* Add deep learning models (BERT, LSTM)
* Deploy on cloud
* Add multilingual support

## Author

**Areen Chauhan**

##  References

* Scikit-learn Documentation
* Flask Documentation
* Research papers on Fake News Detection

---
