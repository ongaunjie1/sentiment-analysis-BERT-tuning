# Sentiment analysis on women'clothing dataset using transformer model (BERT)

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

## 5. What is the distribution of words per review? 
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/d25bdd0e-96d6-493c-8b79-8084367d09c4)
* The max_length is 115 (This information allows us to figure out the max_length of the product reviews, it is crucial for the fine-tuning process)

# Loading pretrained BERT models from huggingface and performing inference with the model.
Testing pretrained BERT models with different output classes without fine-tuning the model to our dataset:
* 1st Model: **nlptown/bert-base-multilingual-uncased-sentiment** = Multi-class classication: between 1 and 5 (5 classes)
* 2nd Model: **cardiffnlp/twitter-roberta-base-sentiment-latest** = Multi-class classification: Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive (3 classes)
* 3rd Model: **distilbert-base-uncased-finetuned-sst-2-english** = Binary classification: 0 -> Negative; 1 -> Positive (2 classes)

| Models                 | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| 5 classes (1st model)  |   0.57   |   0.65    |  0.57   |   0.59   |
| 3 classes (2nd model)  |   0.79   |   0.77    |  0.79   |   0.78   |
| 2 classes (3rd model)  |   0.84   |   0.85    |  0.84   |   0.84   |
* From the table, binary classification achieved the best results with a **83%** accuracy
* The binary classification model is a DistilBERT, a smaller and computationally efficient version of BERT designed with a smaller memory footprint compared to the original BERT model.
* While using a full BERT model might achieve higher accuracy, it's worth noting that DistilBERT with binary outputs still performs better than the other BERT models in this specific context

# Fine-tuning a distilbert-base model (3 classes) with women's clothing dataset.
Reasoning for fine-tuning a distilbert-base model with an output of 3 classes
* Multi-class classification (5 classes) was not selected because it can be challenging for sentiment analysis, as each rating represents a narrow range of sentiment. This approach would require a larger dataset to 
  effectively capture nuanced sentiment.
* Binary classification (2 classes) was not selected because the distribution of the ratings of this dataset is not too imbalanced. Binary classification is more suitable when dealing with a high prevalence of low and 
  high ratings. Additionally, binary classification, being overly simplistic, can lead to a loss of information by forcing ratings into just two categories."
* Using a model with 3 classes outputs will help provide a more detailed insights into the sentiment of the text. This granularity helps to distinguish between completely positive, completely negative, and neutral 
  sentiments, providing richer information.
* Decided to use distilbert-base model instead of a bert-base model because distilbert model is computationally less intensive. This results in faster training and inference times
* The fine-tuned model will be used to compare against a pretrained-bert-base model from huggingface.

## Refer to the fine_tuning notebook for all the steps of the fine-tuning process
* The model was fine-tuned in a google colab environment (utilizing a GPU)
* The fine tune model was trained on the women's clothing dataset.
* The models will be evaluated using a test dataset that has been split from the train_test_split process in google colab as the original dataset cannot be used to evaluate as the fine-tuned model was trained on it. 
* training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

## Results of the model and comparison between a fine-tuned distilbert model (3 clases) and a pretrained bert model (3 classes)
| Model                            | Accuracy | Precision | Recall  | F1 Score |
|----------------------------------|----------|-----------|---------|----------|
| Pretrained model                 |   0.79   |    0.77   |   0.79  |   0.77   |
| Fine-tuned model                 |   0.85   |    0.86   |   0.85  |   0.85   |

* Pretrained model used: cardiffnlp/twitter-roberta-base-sentiment-latest
* Fine-tuned base model: distilbert-base-uncased
* From the table, fine-tune model performs slightly better than the pretrained model across the board. It achieved an accuracy of **85%**
* Confusion matrix for the fine-tuned model:
  ![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/5421963d-2533-4a00-a041-3005c1818ac5)

## Further improvements:
* Continue Fine-Tuning the model with Different Parameters
* Obtain more data and retrain the model
* Train model with different output classes

## Link to the fine-tuned model: https://huggingface.co/ongaunjie/distilbert-cloths-sentiment
**Try out the model by inputting a sentence!**
* Example input: "this dress is kinda okay" 
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/e8fb9679-5991-450d-adcf-cb5b20d100fa)

**The output labels are as follows:**
* 0 - Negative
* 1 - Neutral
* 2 - Positive



  




 








