import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or headline below. The model will tell you if it's **Fake** or **Real**, along with a confidence score.")

# Input from user
user_input = st.text_area("📝 Paste the news content here:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess and predict
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0][prediction] * 100

        # Display results
        if prediction == 1:
            st.success(f"🟢 Real News ({confidence:.2f}% confident)")
        else:
            st.error(f"🔴 Fake News ({confidence:.2f}% confident)")

        st.markdown("---")
        st.markdown("📌 **Note:** Accuracy may drop for short or vague inputs. The model works best with full paragraphs.")
