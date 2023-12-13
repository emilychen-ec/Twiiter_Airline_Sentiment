
![This is an image](https://images.squarespace-cdn.com/content/v1/5ad94ebf9772aeb9d323b26d/1601397168018-1IGN1VOKO61EPI1A3KF3/20140125-IMG_8339.jpg)
# Twitter_Airline_Sentiment_Analysis
Nowadays, billions of people worldwide are already engaged in social media communities. It becomes a primary place for life and opinion sharing. Twitter, as one of the leading social media companies in the world, gives consumers direct access to the brands by adding a simple mention or @ to their tweets. Analyzing tens of thousands of tweets and collecting valuable customer feedback becomes essential to business owners. 

In the US airline industry, a few companies take up a high percentage of the market and can't afford to ignore the actions of others. Therefore, insight into customers' feedback on their business and competitors' business is essential. 

The project used machine learning models to identify the emotional tone the tweets carry and collected the most predictive keywords, which allow the airlines to improve offered services quickly and aid them in devising new schemes for the future.

## **Table of Contents**


1. Data
  - [Original tweets Data](data/Tweets.csv)
  - [Cleaned Data](data/btweets_cleaned.csv)
  - [text_preprocessor.py](data/text_preprocessor.py)
  
  
2. Notebooks
  - [Data Wrangling](notebook/Tweets_Airline_Data_Wrangling.ipynb)
  - [Exploration Data Analysis](notebook/Tweets_EDA.ipynb)
  - [Machine Learning Models](notebook/Tweets_Modeling.ipynb)
  
3. Documents
  - [Final Report](https://docs.google.com/document/d/1SPX8VmHpw5DvKM18gQ2aLTrVkpAotd-QcV0TdZE_MwA/edit?usp=sharing)
    - This report contains a detailed account of the data wrangling, EDA and modeling performed for this project.  
  - [Presentation](https://docs.google.com/presentation/d/1EQYTSFri9Lce5cgyDHnpk018lQ0gAGoHmCMynm_UU28/edit?usp=sharing)
    - This presentation summarizes the data wrangling, EDA, modeling and conclusions for this project.


[Gan introduction](https://www.youtube.com/watch?v=8L11aMN5KY8&ab_channel=Serrano.Academy)
[Convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown)
(edge ditection, choose kernel, flip around the kernel)
[Nick github image with GAN](https://github.com/nicknochnack/GANBasics)
[Gan loss function deeper explanation](https://neptune.ai/blog/gan-loss-functions)
[Application of generative AI](https://www.xenonstack.com/blog/generative-video-models)
[Medical image generation article](https://arxiv.org/ftp/arxiv/papers/2005/2005.10687.pdf)
(p4 DCGAN – prevent model collision, produce better resolution images
**Application**: 1 reconstruction, denoise 2 image synthesis – lack enough sample or need skills to diagnose or cross modality synthesis such as generate CT based on MRI;
The probable reason for the synthesis of MRI images is that it takes longer scan time for multiple sequence acquisition. Conversely, GAN effectively generates the next sequence from the acquired one, which saves time slots for another patient
**Challenges**: it is required to explore the validity of the metrics. Another big challenge is the absence of data fidelity loss in case of unpaired training. unable to retain the information of the minor abnormality region during the cross-domain image-to-image translation process)
