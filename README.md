# 📰 Fake News Detection Using Machine Learning (SVM)

This project is a **Fake News Detection system** built using **Python, NLP, TF-IDF vectorization, and Support Vector Machine (SVM)** classification.  
It classifies news articles as **Real** or **Fake** based on their textual content.

---

## 🚀 Features

- Clean and preprocessed dataset (Fake.csv + True.csv)
- NLP pipeline using:
  - Text normalization
  - Stopword removal
  - Punctuation removal
  - TF-IDF vectorization
- Trained using **LinearSVC**
- Achieves high accuracy on Fake vs Real news classification
- Includes **graphical output** (circle indicator)
  - 🟢 Green Circle → Real News  
  - 🔴 Red Circle → Fake News
- Saved model (`model.pkl`) and TF-IDF vectorizer (`tfidf.pkl`)
- Ready for deployment in Python apps, Flask, Streamlit, or web apps

---

## 📂 Project Structure

Fake-News-Detection/
│── train_model.ipynb
│── test_model.ipynb
│── Fake.csv
│── True.csv
│── model.pkl
│── tfidf.pkl
│── README.md




---

## 📥 Dataset

The dataset used in this project comes from Kaggle:

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It contains:

- **Fake.csv** → Fake news articles  
- **True.csv** → Real news articles  

We added labels manually:

- `0` → Fake  
- `1` → Real  

---

## 🧹 Data Preprocessing

Steps performed:

1. Convert text to lowercase  
2. Remove punctuation  
3. Tokenize text  
4. Remove stopwords  
5. Create a new column: `clean_text`  
6. Convert text to numeric vectors using **TF-IDF**

---

## 🤖 Model Training (SVM)

We use:

```python
model = LinearSVC()
model.fit(X_train_vec, y_train)

After training, we save:

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf.pkl", "wb"))

🔮 Future Improvements

Add deep learning model (LSTM / BERT)

Deploy using Flask or Streamlit

Add web UI for easy testing

Add explainability (LIME / SHAP)



🧑‍💻 Developed By

Developed by **Kanha Patidar**

Branch: B.Tech CSIT

Semester: 5th Sem

College: Chameli Devi Group of Institutions, Indore


GitHub: kanha165

LinkedIn: (https://www.linkedin.com/in/kanha-patidar-837421290/)

Email: (kanhapatidar7251@gmail.com)


 
Project: Fake News Detection using SVM

⭐ Support

If you like this project, please ⭐ star the repository and share it with others.
