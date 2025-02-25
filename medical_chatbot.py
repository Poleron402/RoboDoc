
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# ### Download required NLTK resources

# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt_tab')



# %%
df = pd.read_csv('train_data2.csv')

# %%
df['Description'] = df['Description'].str.replace(r'^Description:\s*', '', regex=True)
df['Description'] = df['Description'].str.replace(r'^Q\.\s*', '', regex=True)

df = df[['Patient', 'Description', 'Doctor']].drop_duplicates()

# %% [markdown]
# ### Analyze data

# %%
class_counts = df['Patient'].value_counts()
print(class_counts.describe())

plt.figure(figsize=(12, 6))
df['Patient'].value_counts().hist(bins=100, edgecolor='black')
plt.yscale('log')  # Log scale to see rare and frequent cases clearly
plt.xlabel('Number of Occurrences per Patient')
plt.ylabel('Frequency (log scale)')
plt.title('Distribution of Patient Occurrences')

# %% [markdown]
# ### Enhanced cleaning: lowercase, remove prefixes, numbers, special characters, lemmatize, and remove stopwords.
# 

# %%
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english')).union({
    'hi', 'hello', 'doctor', 'year', 'old', 'thanks', 'yrs',
    'years', 'patient', 'patients', 'dear', 'sir', 'thank',
    'age', 'name', 'doc'
    })

# %% [markdown]
# ### Compile regex patterns

# %%
number_pattern = re.compile(r'\d+')
special_char_pattern = re.compile(r'\W+')

# %% [markdown]
# ### Text Preprocessing Function

# %%
def enhanced_clean_text(text):
    text = text.lower()
    text = re.sub(r'patient:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'description:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens)

df['Cleaned_Patient'] = df['Patient'].apply(enhanced_clean_text)
df['Cleaned_Description'] = df['Description'].apply(enhanced_clean_text)

# %% [markdown]
# ### Label encode the cleaned descriptions as our target classes.

# %%
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Cleaned_Description'])

# %% [markdown]
# ### Filter out classes with fewer than 5 samples to reduce the number of target classes.
# 

# %%
label_counts = df['Label'].value_counts()
min_samples = 5
valid_labels = label_counts[label_counts >= min_samples].index
df_filtered = df[df['Label'].isin(valid_labels)]

# %% [markdown]
# ### Split Data

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered['Cleaned_Patient'],
    df_filtered['Label'],
    test_size=0.2,  # adjust if needed so that the test set sample count >= number of classes
    random_state=42,
    stratify=df_filtered['Label']
)

# %% [markdown]
# ### TF‑IDF vectorization with n-grams and tuned frequency thresholds.
# 

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %% [markdown]
# ### Use Logistic Regression with hyperparameter tuning via GridSearchCV.
# 

# %%
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}
lr = LogisticRegression(max_iter=500, solver='lbfgs')
grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_tfidf, y_train)
best_lr = grid.best_estimator_


# %%
y_pred = best_lr.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Revised Classical Model Accuracy: {accuracy * 100:.2f}%")


# %%
import en_core_web_sm
nlp = en_core_web_sm.load()


# %%
def extract_entities(text):
    doc = nlp(text)
    # Attempt to extract using spaCy NER for custom labels (likely none with en_core_web_sm)
    spacy_symptoms = [ent.text for ent in doc.ents if ent.label_ in ["SYMPTOM", "DISEASE"]]

    # Define a set of common symptom keywords as a fallback.
    symptom_keywords = {"fever", "cough", "headache", "nausea", "dizziness", "fatigue", "pain", "sore", "throat"}
    tokens = word_tokenize(text.lower())
    rule_based_symptoms = [token for token in tokens if token in symptom_keywords]

    # Combine both methods
    all_symptoms = set(spacy_symptoms + rule_based_symptoms)
    return ", ".join(all_symptoms) if all_symptoms else "No symptoms detected"

# %%
def chatbot():
    st.title("🩺 Medical Chatbot")
    st.write("Enter your symptoms, and I'll suggest possible conditions.")

    user_input = st.text_area("Describe your symptoms:")

    if st.button("Get Diagnosis"):
        if user_input:
            # Preprocess input: clean and vectorize the text
            cleaned_input = enhanced_clean_text(user_input)
            input_vector = vectorizer.transform([cleaned_input])

            # Predict the condition cluster using the trained model
            prediction_cluster = best_lr.predict(input_vector)[0]
            prediction = f"Cluster {prediction_cluster}"

            # Extract relevant symptoms using the hybrid NER function
            extracted_symptoms = extract_entities(user_input)

            # Retrieve doctor's advice associated with this cluster if available.
            # Here we use a placeholder response; adjust if you have an advice mapping.
            response = "No doctor advice available for this prediction."

            # Display results in Streamlit
            st.subheader("Possible Condition:")
            st.write(prediction)

            st.subheader("Symptoms Detected:")
            st.write(extracted_symptoms)

            st.subheader("Doctor's Advice:")
            st.write(response)
        else:
            st.write("Please enter your symptoms.")

if __name__ == "__main__":
    chatbot()


