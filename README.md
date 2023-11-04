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

model = Model(X, y, LogisticRegression(C=best_params_lr['classifier__C'], max_iter=best_params_lr['classifier__max_iter']),   
              CountVectorizer(preprocessor=preprocess, max_features=best_params_cv['count_vectorizer__max_features'],  
 ngram_range=best_params_cv['count_vectorizer__ngram_range']))  



**Source :**   
https://www.kaggle.com/datasets/kazanova/sentiment140/  
https://openclassrooms.com/fr/courses/5801891-initiez-vous-au-deep-learning/5814656-decouvrez-les-cellules-a-memoire-interne-les-lstm  
https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizerModel.html  
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html  
https://www.kaggle.com/code/arunrk7/nlp-beginner-text-classification-using-lstm  

**Comments :**  
The pre processing part, model training and model improvement was very well understood. I had a lot more trouble with the Pytorch and LSTM part, I took a lot from the internet and I had trouble completely understanding all the steps and the logic behind it.  



  # ***Build your own NLP Project***

This in your occasion to work on an NLP project from the **exploratory data analysis** (EDA) **to the inference phase**, on a topic of your choice. Don't be intimadated by the number of steps, there's almost nothing new compared to previous assignments!

## ***1. Select a dataset***

Find a text dataset on the topic you're interested in, websites like [**Kaggle**](https://www.kaggle.com/datasets) provide ready-to-use datasets. You can choose a task we've already solved in class (**classification** or **sentiment analysis**) or a new one if you feel ready!

## ***2. Exploratory data analysis $⇾$ jupyter notebook***

Start by exploring your data, examine some samples, plot the information that you deem relevant, identify potential features you can engineer to improve performance later ...

## ***3. Build your preprocessing pipeline $⇾$ python script***

Similarly to the first lab, assemble a preprocessing script. Make sure to explain your design choices and why you use specific functions in a particular order. You can also add any utility function that you might need later to this script.

## ***4. Train a baseline model $⇾$ jupyter notebook***

Import your preprocessing pipeline, apply it to your dataset and train the machine learning model of your choice (sklearn or similar) without any particular parameter tuning or feature engineering. The goal here is simply to obtain a baseline model which we'll use as reference for future experiments. This is the good moment to create a model class (remember lab 2) that will facilitate iteration later on.

## ***5. Improve on the baseline $⇾$ jupyter notebook***

Using techniques of your choice, improve on the baseline results. This is the moment to demonstrate your ability to identify the bottlenecks in your training process and potential problems with your model / data.

Here are some common ideas to get you started:

* Balancing classes (oversampling or subsampling)
* Applying class weights
* Monitoring training curves
* Early stopping
* Hyperparameter tuning, grid search, random search
* Or anything else you deem coherent

## ***6. Use Tensorflow (or PyTorch, JAX, ...) and train a sequence model of your choice (RNN, GRU, LSTM, Transformer, ...) $⇾$ jupyter notebook***

Finally, we want to further improve the performance of our model by applying our deep learning skills to the problem. You can either build and train a model from scratch or finetune a pre-trained model (transfer learning, you can find models to download on websites like [***HuggingFace***](https://huggingface.co/models)). Make sure to comment on the architecture you decide to use!

## ***7. Publish your project on GitHub and add a nice README.md file***

Now that your project is complete, publish it on your GitHub and add a README.md file (it will be used as the landing page of your repository). This readme should contain:

* A global description of your project
* A link to your dataset
* A table containing the performances of each model you implemented
* Instruction on how to install and run your project (you can use poetry, anaconda, or a requirements.txt file)
* A list of references that you used (coding tutorials, research papers, other documents ...)
