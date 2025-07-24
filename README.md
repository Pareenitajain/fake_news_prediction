## 📰 Fake News Detection using NLP

This project aims to detect whether a news article is **fake** or **real** using Natural Language Processing (NLP) techniques and machine learning. It uses a dataset of real and fake news headlines and applies TF-IDF vectorization along with a Logistic Regression classifier.

---

### ✅ Project Goals

* Classify news articles as **Real** or **Fake**
* Preprocess and vectorize text using **TF-IDF**
* Train a **Logistic Regression** model
* Evaluate the model performance using metrics like **accuracy, precision, recall, and F1-score**
* Build a **Streamlit web app** for real-time predictions

---

### 📂 Dataset

* **Source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **Files Used:** `Fake.csv` and `True.csv`
* Each news article is labeled:

  * `0` → Fake News
  * `1` → Real News

---

### 🛠️ Technologies & Tools

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK for text preprocessing
* TF-IDF Vectorizer
* Logistic Regression Classifier
* Streamlit (for deployment)

---

### 🔍 Model Performance

* **Accuracy:** 98.91%
* **F1-score:** 0.99
* The model performs well in distinguishing fake and real news on the test set.

---

### 🧪 Sample Output

```
Prediction: Fake
Confidence: 68.04%
Note: Accuracy may drop for short or vague inputs.
```

---

### 🚀 How to Run the Streamlit App

1. Clone the repository
2. Install requirements:

   ```
   pip install -r requirements.txt
   ```
3. Run the app:

   ```
   streamlit run app.py
   ```

---

### 📌 Limitations

* The model is based on **TF-IDF**, so it may not capture deep semantic meaning.
* It may misclassify **real news with vague or short text**.

---

### 📈 Future Improvements

* Fine-tune a **BERT model** for better semantic understanding.
* Improve UI of the web app.
* Add feature for news source verification.

---

### 🙋‍♀️ Author

Pareenita Jain

