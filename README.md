# Sentiment Analysis on clothing's review dataset using transformer models (BERT)

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

# Project Goals
## 1. Data Cleaning and EDA
Perform data cleaning and exploratory data analysis (EDA) on the dataset to uncover insights from product reviews.
## 2. Utilizing Hugging Face's Pretrained Models
Utilize Hugging Face's pretrained models to predict customer sentiments based on product reviews. This involves leveraging state-of-the-art transformer-based models like BERT for sentiment analysis.
## 3. Testing Different BERT Models (Without Fine-Tuning)
Test different types of BERT models from Hugging Face with varying output classes. This step involves experimenting with pretrained models to evaluate their performance without fine-tuning.
## 4. Decision on Number of Output Classes for Final Model
Make a decision on the number of output classes for the final sentiment analysis model. 
## 5. Fine-tune the BERT model to the dataset
Fine-tune the dataset with the decided number of output classes

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

# Types of BERT models used in this project:
| Feature                  | BERT                                    | RoBERTa                                  | DistilBERT                              |
|--------------------------|-----------------------------------------|------------------------------------------|-----------------------------------------|
| **Training Objectives**  | MLM (Masked Language Model), NSP        | MLM (Masked Language Model)              | MLM (Masked Language Model)             |
| **Data Preprocessing**   | Random Masking                          | Dynamic Masking, No NSP                  | Random Masking, Pruning Attention       |
| **Next Sentence Prediction (NSP)** | Yes                           | No                                       | No                                      |
| **Training Duration**    | Extended                                | Longer, Larger Dataset                   | Shorter, Pruned Layers                  |
| **Sentence Embeddings**  | [CLS] Token                             | No [CLS] Token for Sentence Tasks        | [CLS] Token                             |
| **Batch Training**       | Fixed Batch Size                        | Dynamic Batch Size                       | Smaller Model Size                      |
| **Model Size**           | Large                                   | Larger                                   | Smaller                                 |
| **Number of Layers**     | Configurable, Typically 12 or 24        | Configurable, Typically 12 or 24         | Reduced (Distilled), Typically 6        |
| **Performance**          | Benchmark Model                         | Improved Performance on Tasks            | Trade-Off between Size and Quality      |

# Loading pretrained BERT models from huggingface and performing inference with the models.
Testing different types of pretrained BERT models with different output classes without fine-tuning the models to our dataset:
* BERT model: Pretrained on 5 output classes  (1 star to 5 star) - [Link to the model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
* roBERTa model: Pretrained on 3 output classes (0 : Negative, 1 : Neutral, 2 : Positive) - [Link to the model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
* distilBERT model: Pretrained on 2 output classes (0 : Negative, 1 : Positive) - [Link to the model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

# Results of the model
| Models                 | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| BERT model (5 output classes)  |   0.57   |   0.65    |  0.57   |   0.59   |
| roBERTa model (3 output classes)  |   0.79   |   0.77    |  0.79   |   0.78   |
| distilBERT (2 output classes)  |   0.84   |   0.85    |  0.84   |   0.84   |
* Even without fine-tuning, the distilBERT already achieved a relatively good accuracy of 83% with 2 output classes
* While the roBERTa model also achieved a pretty high accuracy wth an output of 3 classes
* As expected, the BERT model with 5 output classes performed the worst due to its narrow sentiment range.

# Fine-tuning distilbert-base with an 3 output classes
Reasons for choosing to fine-tune a distilBERT model with an output of 3 classes:
* Multi-class Classification (5 classes): Avoided due to the dataset's narrow sentiment ranges, requiring a larger dataset for effective capture.
* Binary Classification (2 classes): Not chosen as the dataset's rating distribution is relatively balanced; binary classification risks oversimplifying and losing information.
* 3 Classes for Detailed Insights: Chose 3 classes to distinguish between positive, negative, and neutral sentiments, providing richer insights.
* Choice of DistilBERT over BERT and roBERTa: Selected distilbert-base for faster training and inference times, maintaining computational efficiency.

## Refer to the fine_tuning notebook for all the steps of the fine-tuning process
* The model was fine-tuned in a google colab environment (utilizing a GPU)
* The fine tune model was trained on the women's clothing dataset.
* The models will be evaluated using a test dataset that has been split from the train_test_split process in google colab as the original dataset cannot be used to evaluate as the fine-tuned model was trained on it.
### Training arguments used for fine-tuning:
```
   training_args = TrainingArguments(
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
```

## Results of the fine-tuned distilbert model (3 classes) and the pretrained-base model (3 classes)
| Model                            | Accuracy | Precision | Recall  | F1 Score |
|----------------------------------|----------|-----------|---------|----------|
| Pretrained-base model            |   0.79   |    0.77   |   0.79  |   0.77   |
| Fine-tuned model                 |   0.85   |    0.86   |   0.85  |   0.85   |

* Pretrained model used: cardiffnlp/twitter-roberta-base-sentiment-latest
* Fine-tuned base model: distilbert-base-uncased
* From the table, fine-tune model performs slightly better than the pretrained model across the board. It achieved an accuracy of **85%**
* Confusion matrix for the fine-tuned model:
  ![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/5421963d-2533-4a00-a041-3005c1818ac5)

## Further improvements:
* Continue Fine-Tuning the model with Different Parameters
* Obtain more data and retrain the model
* Fine-tune the model with different output classes

# Link to the fine-tuned model: https://huggingface.co/ongaunjie/distilbert-cloths-sentiment

## 1) Try out the model by inputting a sentence

![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/1c809e53-5dac-4e46-b92c-cf2c8368d2d9)
* Example input: "this dress is kinda okay"
### The output labels are as follows:
* 0 - Negative
* 1 - Neutral
* 2 - Positive


## 2) Perform inference on the model, more details can be found in the huggginface's repository**
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/eca34d97-6c9f-4cc3-8e71-b58db9716ffe)




  




 








