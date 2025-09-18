import streamlit as st
import pickle
import nltk
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Pre-processing
def transform_text(text):
    # Converting text to lower case
    text=text.lower()
    # Tokenizing text to words
    text=nltk.word_tokenize(text)
    
    # Removing special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    # Removing stop words and punctuations
    for i in text:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    # Stemming
    from nltk.stem.porter import PorterStemmer
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
mnb=pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Classifier')

input_sms=st.text_input('Enter the sms: ')

if st.button('Predict'):
    # 1. Pre-processing text
    transformed_sms=transform_text(input_sms)

    # 2. Vectorization
    vectorized_sms=tfidf.transform([transformed_sms])

    # 3. Prediction
    result=mnb.predict(vectorized_sms)[0]

    # 4. Display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')