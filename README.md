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

**Main goal of this project:**
* 1st: Performing data cleaning and EDA on the dataset to uncover insights from the product reviews.
* 2nd: Utilizing huggingface's pretrained models to predict customer sentiments based on product reviews.
* 3rd: Fine-tune a base BERT model and compare its performance with a pretrained model from huggingface.

# Data cleaning
* Removed null values
* Removed duplicated rows

# EDA (Exploratory Data Analysis)
Answering the following questions
## How many product reviews per sentiment?
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/4992b6e9-9d42-4c02-acb3-18262d5b87cf)


