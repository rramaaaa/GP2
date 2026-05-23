# FakeNewsProject

## Dataset
The dataset used in this project is AFND (Arabic Fake News Dataset), Dataset link: https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd

### Dataset Description
The (AFND) is a large-scale Arabic news dataset that was built specifically for the purpose of fake news detection. It contains more than 600K Arabic news articales that were collected from 134 different Arabic news websites across 19 Arabic countries.
Each artical in the dataset is label based on how credible its source website is. The dataset includes articles from both credible and non-credible sources.

### Data Cleaning
Since the data consists of Arabic text collected from various news websites, it contained a lot of noise that needed to be handled fisrt.
- Removing URLs and email addresses (these add no useful information for classification)
- Normalizing Arabic characters
- Removing diacritics (not needed for understanding the meaning of a word in this context)
- Removing non-Arabic characters
- Removing extra whitespace
- Remove duplicate articles
### Feature Construction
The title and article text were combined into a single feature called content.

content = title + text

This allows the model to analyze both the headline and the full article together, providing more context for detecting fake news.
