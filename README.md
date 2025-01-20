****Twitter Sentiment Analysis using NLP****

**Overview**

This project predicts tweet sentiment (positive or negative) using a machine learning model with Natural Language Processing (NLP).

**Features**

This project uses a dataset of 1.6 million tweets from Kaggle, labeled as positive (`1`) or negative (`0`). Key techniques include data cleaning, stemming, and stop word removal to preprocess text for analysis.

**Importing Data**

The dataset is retrieved using the Kaggle API, downloaded as a ZIP file, and extracted to obtain a CSV for analysis.

**Data Processing**

The data undergoes the following processing steps:
- Renamed columns for readability and verified no missing values.
- Balanced dataset: 800K positive and 800K negative tweets.
- Re-encoded positive labels from `4` to `1` for consistency.
- Cleaned text by removing special characters, numbers, and stop words, converting to lowercase, and splitting into word lists.
- Applied stemming with PorterStemmer to reduce words to root forms (e.g., "acting" to "act"), creating a new `stemmed_content` column.

**Data Preparation**

Features (`X`) are extracted from the `stemmed_content` column, and targets (`Y`) are derived from the `target` column. The dataset is split into 80% training and 20% testing sets, with stratification to maintain label distribution.

```python
from sklearn.model_selection import train_test_split
X = tweets_data['stemmed_content'].values
Y = tweets_data['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```





