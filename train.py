#train models as per parameters
#updates the saved model files
#called only during the training process

def trainer():
    
    print("Inside train.py file")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.ensemble import RandomForestClassifier
    from textpreprocessor import tpp
    from nltk.stem.wordnet import WordNetLemmatizer
    seed = 2510 
    true_data = pd.read_csv('datasets/True.csv')
    fake_data = pd.read_csv('datasets/Fake.csv')
    true_data['label']=1
    fake_data['label']=0

    news_data = pd.concat([true_data, fake_data], axis=0)
    news_data['fulltext'] = news_data.title + ' ' + news_data.text
    news_data.drop(['title','text'], axis=1, inplace=True)
    news_data = news_data[['fulltext', 'label']]

    print('Received {} rows and {} columns as input.'.format(news_data.shape[0], news_data.shape[1]))
    x_fulltext=news_data['fulltext']
    x_fulltext = x_fulltext.reset_index().drop(['index'], axis=1)
    x_fulltext = x_fulltext.values.tolist()


    #calling tpp function from textprocessor
    x_fulltext_tokenized=tpp(x_fulltext)
         
    # function for lemmatizing
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(x_data):
        lemma_list=[]
        for token in x_data:
            words = lemmatizer.lemmatize(token)
            lemma_list.append(words)
        return lemma_list
    
    x_fulltext_lemmatized = []
    for i in range(len(x_fulltext)):
        test_data = lemmatize_words(x_fulltext_tokenized[i])
        x_fulltext_lemmatized.append(test_data)
        
    cv = CountVectorizer(max_features=1000)
    a=[]
    for x in x_fulltext_lemmatized:
        a.append(' '.join(x))
        

    x_data_full_vector = cv.fit(a)
    x_data_full_vector = cv.transform(a).toarray()
    tfidf = TfidfTransformer()
    x_data_full_tfidf = tfidf.fit_transform(x_data_full_vector).toarray()
    y_data=news_data['label']
    X_train, X_test, y_train, y_test = train_test_split(x_data_full_tfidf, y_data, test_size=0.20, random_state= seed)

# ### Multinomial Niave Bayes

# In[ ]:


# Instatiation, fitting and prediction

    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)
    predictions = MNB.predict(X_test)


# In[ ]:


# Model evaluation

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    MNB_f1 = round(f1_score(y_test, predictions, average='weighted'), 3)
    MNB_accuracy = round((accuracy_score(y_test, predictions)*100),2)

    print("Accuracy : " , MNB_accuracy , " %")
    print("f1_score : " , MNB_f1)


    # ### Random Forest

    # In[ ]:


    # Instatiation, fitting and prediction

    rfc=RandomForestClassifier(n_estimators= 10, random_state= seed)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)


    # In[ ]:


    # Model evaluation

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    rfc_f1 = round(f1_score(y_test, predictions, average= 'weighted'), 3)
    rfc_accuracy = round((accuracy_score(y_test, predictions) * 100), 2)

    print("Accuracy : " , rfc_accuracy , " %")
    print("f1_score : " , rfc_f1)

trainer()

