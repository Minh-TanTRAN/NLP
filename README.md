# NLP

About Dataset
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

**Content**
It contains the following 6 fields:

- target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

- ids: The id of the tweet ( 2087)

- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

- flag: The query (lyx). If there is no query, then this value is NO_QUERY.

- user: the user that tweeted (robotickilldozr)

- text: the text of the tweet (Lyx is cool)


    
**Some of the common text preprocessing that we use:**
* Lower casing
* Removal of Punctuations
* Removal of Stopwords
* Stemming
* Lemmatization
* Removal of URLs 
* Removal of HTML tags

  
 
**Results :**    
  

              precision    recall  f1-score   support    
    Negative       0.81      0.76      0.78    159494  
    Positive       0.77      0.82      0.80    160506  
  
    accuracy                           0.79    320000  
    macro avg       0.79      0.79      0.79    320000   
    weighted avg       0.79      0.79      0.79    320000      


model = Model(X, y, LogisticRegression(),     
              CountVectorizer()  )  with best parameters by GridSearchCV()  



  

**Source :**   
https://www.kaggle.com/datasets/kazanova/sentiment140/  
https://openclassrooms.com/fr/courses/5801891-initiez-vous-au-deep-learning/5814656-decouvrez-les-cellules-a-memoire-interne-les-lstm  
https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizerModel.html  
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html  
https://www.kaggle.com/code/arunrk7/nlp-beginner-text-classification-using-lstm  


    
**Conclusion**    
During this project, we used the Sentiment140 Dataset. On this Dataset, we first performed preprocessing on the data to make the dataset suitable for learning algorithms. We have removed capital letters, removed punctuation, emails and http sites.
  
Next, we tested the predictions of the Sentiment class on our first pipeline model. We then sought to improve the pipeline by testing different classifier,models. When we found our best model, we used GridSearchCv to find the best hyperparameters.
  
The last file constitutes the Deep Learning model where we used the LTSM model.

    
**Acknowledgments**  
- Kaggle for the dataset
- NLP courses from Ryan Pegoud
    


