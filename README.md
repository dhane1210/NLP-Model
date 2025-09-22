Mental Health Text Classification (NLP Project)

This project focuses on **detecting mental health conditions from Reddit posts** using **Natural Language Processing (NLP)** and a fine-tuned **BERT model**.  

The pipeline covers:
1. **Data Collection** → Scraping Reddit posts from mental health–related subreddits.  
2. **Data Processing** → Cleaning, balancing, and preparing the dataset.  
3. **Model Training** → Fine-tuning a BERT model for multi-class text classification.  
4. **Deployment** → Exposing the model via a Flask REST API.  

Repository Structure

NLP-Model/
│
├── 01_Reddit_Data_Scraping.ipynb # Notebook for scraping & cleaning Reddit posts
├── 02_Model_Training_Evaluation.ipynb # Notebook for training & evaluating MentalBERT
├── mental_health_balanced.csv # Final balanced dataset used for training
├── app.py # Flask backend API for inference


Dataset

- **Source**: Reddit posts from multiple subreddits (e.g., `depression`, `Anxiety`, `PTSD`, `SuicideWatch`, etc.).  
- **Classes**: 12 mental health conditions + "Normal".  
- **Balancing**: Oversampling/undersampling applied to ensure ~1000 samples per class.  
- **File**: `mental_health_balanced.csv`


Notebooks

Reddit Data Scraping (`01_Reddit_Data_Scraping.ipynb`)
- Used **PRAW (Python Reddit API Wrapper)** to scrape posts.  
- Preprocessed text (removed URLs, mentions, hashtags, emojis, etc.).  
- Collected ~2000 posts per condition.  
- Exported raw dataset → `mental_health_dataset.csv`.  

Model Training & Evaluation (`02_Model_Training_Evaluation.ipynb`)
- Used **HuggingFace Transformers** (`mental/mental-bert-base-uncased`).  
- Tokenized text and fine-tuned for classification.  
- Balanced dataset used: `mental_health_balanced.csv`.  
- Performed:
  - Class distribution analysis  
  - Text length statistics  
  - Word frequency exploration  
  - Confusion matrix & ROC curves  
  - Error analysis (short text, sarcasm, code-mixing)  
- Final model saved as **`mental_bert_model/`**.  

---

Flask API (`app.py`)

The trained model is deployed with **Flask** for easy integration into applications.  
