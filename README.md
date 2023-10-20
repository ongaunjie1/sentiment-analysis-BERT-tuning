# Sentiment analysis on women'clothing using transformer model (BERT)

**Dataset informations:**
* Clothing ID: A unique identifier for each clothing item.
* Age: The age of the reviewer.
* Title: The title of the review provided by the reviewer.
* Review Text: The detailed text of the review where the reviewer expresses their thoughts and opinions about the product.
* Rating: The rating given by the reviewer, typically on a scale from 1 to 5 stars.
* Recommended IND: An indicator (usually binary) that suggests whether the reviewer recommended the product (e.g., 1 for recommended, 0 for not recommended).
* Positive Feedback Count: The count of positive feedback or "likes" received on the review by other users.
* Division Name: The name of the product division or category.
* Department Name: The name of the department within the division to which the product belongs.
* Class Name: The specific class or category to which the product belongs.

**Main goals of this project:**
* 1st: Performing data cleaning and EDA on the dataset to uncover insights from the product reviews.
* 2nd: Utilizing huggingface's pretrained models to predict customer sentiments based on product reviews.
* 3rd: Fine-tune a base BERT model to our dataset and compare its performance with a pretrained model from huggingface.

# Data cleaning
* Removed null values
* Removed duplicated rows

# EDA (Exploratory Data Analysis)
Answering the following questions
## 1. How many product reviews per rating?
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/4992b6e9-9d42-4c02-acb3-18262d5b87cf)
* From the graph, it shows that the ratings are increasing in a linear trend.
* While there is some imbalance, it doesn't seem extreme, as there is a reasonable spread of ratings across all values from 1 to 5 stars.

## 2. What is the number of reviews per product category? 
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/4d78f622-edab-4bab-9ab6-d1b8d36161fc)
* From this dataset, it seems that dresses, knits and blouses are among the highest reviewed products.

## 3. What is the distribution of reviews per rating per category?
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/68a4b08e-c6c9-45eb-910f-e66e6f4a0935)
* From the plot, all the products are fairly balanced with its distribution of ratings.

## 4. Ploting a word cloud for the product reviews
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/e3cf1137-e517-482f-9e42-f9f24d3e2e0e)

## 5. What is the distribution of words per review? (This information allows us to figure out the max_length of the product reviews, it is crucial for the fine-tuning process)
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/d25bdd0e-96d6-493c-8b79-8084367d09c4)
* The max_length is 115

# Loading pretrained BERT models from huggingface and performing inference with the model.
Testing pretrained BERT models with different output classes without fine-tuning the model to our dataset:
* 1st Model: **nlptown/bert-base-multilingual-uncased-sentiment** = Multi-class classication: between 1 and 5 (5 classes)
* 2nd Model: **cardiffnlp/twitter-roberta-base-sentiment-latest** = Multi-class classification: Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive (3 classes)
* 3rd Model: **distilbert-base-uncased-finetuned-sst-2-english** = Binary classification: 0 -> Negative; 1 -> Positive (2 classes)

| Models                 | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| 5 classes (1st model)  | 0.566092 | 0.653298  | 0.566092| 0.592059 |
| 3 classes (2nd model)  | 0.793413 | 0.771015  | 0.793413| 0.776169 |
| 2 classes (3rd model)  | 0.837224 | 0.849638  | 0.837224| 0.841782 |
* From the table, binary classification achieved the best results with a **83%** accuracy
* The 3rd model is DistilBERT, a smaller and computationally efficient version of BERT designed with a smaller memory footprint compared to the original BERT model.
* While using a full BERT model might achieve higher accuracy, it's worth noting that DistilBERT with binary outputs still performs better than the other BERT models in this specific context

# Fine-tuning a distilbert-base model (3 classes) with our dataset.
Reasoning for choosing a distilbert-base model with 3 clases
* Using a 5-star rating system for sentiment analysis can be challenging, especially when each star rating represents a narrow range of sentiment. It can lead to a fine-grained classification task
  that requires a larger dataset to effectively capture the nuances of sentiment.

 








