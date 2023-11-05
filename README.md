# Sentiment-Analyser

## Dataset

The dataset used was [amazon fine food reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) with a text field and the rating for each reviews and was stored in the /data directory. This was converted to 3 categories i.e., happy, neutral and sad. The dataset contains a huge number of positive reviews compared to negative ones. So, it was undersampled to overcome class imbalance.

## Model

The text was preprocessed to remove stopwords and html tags. Stemming was done using porter stemmer and the stemmed words were vectorised using tf-idf vectorizer. Using bigrams helped in increasing the accuracy. This was then passed to a naive bayes model which gave an acceptable accuracy on the test set. An LSTM model will be used in the future.

## Deployment
