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
1. Perform data cleaning and exploratory data analysis (EDA) on the dataset to uncover insights from product reviews.
2. Test different types of BERT models from Hugging Face with varying output classes. This step involves experimenting with pretrained models to evaluate their performance on the dataset without fine-tuning.
3. Make a decision on the number of output classes (2 classes, 3 classes or 5 classes)  and the type of BERT model to use (BERT, roBERTa or distilBERT) for the final sentiment analysis model 
4. Fine-tune the dataset after deciding on which type of BERT model to use and how many output classes for the final model.

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

# Results of the pretrained models
| Models                 | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| pretrained BERT model (5 output classes)  |   0.566   |   0.653    |  0.566   |   0.592   |
| pretrained roBERTa model (3 output classes)  |   0.793   |   0.771    |  0.793   |   0.776   |
| pretrained distilBERT (2 output classes)  |   0.837   |   0.850    |  0.837   |   0.842   |
* As expected, the lower output classes will have an easier learning pattern. Hence higher accuracy.
* The BERT model with 5 output classes performed the worst due to its narrow sentiment range.
* Even without fine-tuning, the distilBERT already achieved a pretty good accuracy of 84% with 2 output classes
* In comparison, the roBERTa model also achieved a relatively high accuracy of 79% eventhough it is predicting 3 classes, only a 0.5% difference between roBERTA and distilBERT. This is also expected, because roBERTa has the largest number of parameters among the three.

# Decision on which type of BERT model to use
* For this project, the choice will be distilBERT over BERT and roBERTa because distilBERT has a faster performance in both training and inference times. DistilBERT's smaller size and streamlined architecture contribute to quicker computations, ensuring computational efficiency throughout the model's lifecycle.
* [Link to the base-distilBERT model](https://huggingface.co/distilbert-base-uncased)

# Decision on how many output classes to use
Reasons for choosing 3 output classes:
* Multi-class Classification (5 classes): Avoided due to the dataset's narrow sentiment ranges, requiring a larger dataset for effective capture.
* Binary Classification (2 classes): Not chosen as the dataset's rating distribution is relatively balanced; binary classification risks oversimplifying and losing information.
* Chose 3 classes to distinguish between positive, negative, and neutral sentiments, providing richer insights.

## Refer to the fine_tuning notebook for all the steps of the fine-tuning process
* The model was fine-tuned in a google colab environment (utilizing a GPU)
* The fine-tuned model was trained on the clothing dataset
* The fine-tuned model was evaluated using a test dataset that has been split from the train_test_split process
* The fine-tuned model was compared with the pretrained roBERTa model with 3 output classes.
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

## Results of the fine-tuned distilBERT model and the pretrained roBERTa model 
| Model                            | Accuracy | Precision | Recall  | F1 Score |
|----------------------------------|----------|-----------|---------|----------|
| pretrained roBERTa  (3 classes)   |   0.789   |    0.772   |   0.789  |   0.773   |
| pretrained distilBERT (2 classes)  |   0.837   |   0.850    |  0.837   |   0.842   |
| Fine-tuned distilBERT model   (3 classes)    |   0.849   |    0.860   |   0.849  |   0.853   |

* Pretrained model used: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
* Fine-tuned base model: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
* From the table, the fine-tuned distilBERT model showed a slight performance improvement compared to the pretrained distilBERT with 2 output classes. This improvement is noteworthy, considering the expectation that having 3 output classes could potentially lead to a lower accuracy. The fine-tuning process allows the model to adapt more closely to the nuances of the specific sentiment analysis task, resulting in enhanced performance.
* Also, the pretrained roBERTa model demonstrates competitive performance, closely trailing the fine-tuned distilBERT, even without undergoing the fine-tuning process. This result aligns with the expectation that roBERTa, with its larger number of parameters and advanced architecture, has the potential for strong out-of-the-box performance. 
* Thus, fine-tuning the roBERTa model could present an opportunity to surpass the performance of the fine-tuned distilBERT.

# Confusion matrix for the fine-tuned model:
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/5421963d-2533-4a00-a041-3005c1818ac5)

## Future improvements:
* Continue Fine-Tuning the distilBERT model with different parameters to achieve a higher accuracy
* Fine-tune a roBERTa base model to the dataset and compare it with distilBERT

# If you are interested in the fine-tuned distilBERT model:
## Link to the fine-tuned model: https://huggingface.co/ongaunjie/distilbert-cloths-sentiment

## 1) Try out the model by inputting a sentence
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/1c809e53-5dac-4e46-b92c-cf2c8368d2d9)
* Example input: "this dress is kinda okay"
### The output labels are as follows:
* 0 - Negative
* 1 - Neutral
* 2 - Positive


## 2) Perform inference on the model, more details can be found in the huggginface's repository**
![image](https://github.com/ongaunjie1/Sentiment-analysis-BERT-tuning/assets/118142884/eca34d97-6c9f-4cc3-8e71-b58db9716ffe)




  




 








