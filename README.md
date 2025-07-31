Readme.md file data
# 📰 Fake News Detection using NLP & Machine Learning
# ➡️ Here is the direct Kaggle link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## 📘 Project Overview
This project tackles the growing issue of **misinformation** by leveraging **Natural Language Processing (NLP)** and **Machine Learning** to detect fake news articles. By analyzing text patterns in news content, the model can predict whether an article is **real or fake**, helping users and organizations combat disinformation online.

---

## 🎯 Objective
To build a machine learning pipeline that can:
- Automatically classify news articles as **fake** or **real**
- Provide a clean and interactive **user interface** for prediction using **Streamlit**
- Offer **insightful visualizations and text patterns** for better understanding

---

## 📊 Dataset Description
- **Files Used**: `fake.csv` and `true.csv`
- **Features**:
  - `title`: Headline of the news article
  - `text`: Full content of the article
  - `subject`: Topic category
  - `date`: Published date
- **Target**: Binary label indicating whether the news is fake (`0`) or real (`1`)

---

## 🛠️ Tools & Libraries Used
- 🐍 Python
- 📚 Pandas, NumPy
- 🧹 NLTK (Lemmatizer, Stopwords, Tokenization)
- 🛠 Scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Train/Test Split, Evaluation)
- 📊 Matplotlib, Seaborn (EDA & Visualization)
- 🌐 Streamlit (Web App Deployment)
- 💾 Pickle (Model Saving/Loading)

---

## 🧹 Text Preprocessing Steps
- Lowercasing
- Removing URLs, punctuation, HTML tags
- Removing stopwords
- Lemmatization using `WordNetLemmatizer`
- Tokenization using NLTK

---

## 🤖 Machine Learning Models
Trained and compared the following:
- **Logistic Regression**
- **Multinomial Naive Bayes**

**Evaluation Metrics:**
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1)

---

## 🧠 Key Insights
- Fake news uses **sensational or exaggerated** vocabulary
- Real news tends to include **named sources** and **formal structure**
- TF-IDF feature extraction proved highly effective in this classification task

---

## 🧪 Results
| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 0.96     |
| Multinomial Naive Bayes| 0.95     |

---

## 💻 Streamlit Web App
An interactive web app was built using **Streamlit** where users can paste any article text and classify it as **real or fake**.

### 🔹 Features:
- Clean interface
- Instant prediction
- Text preprocessing under the hood

### 🟢 To Run Locally:
```bash
pip install streamlit
streamlit run app.py
