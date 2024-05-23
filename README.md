# Predicting-Stock-Price-Changes-Using-News-with-Gemini
Predicting Stock Price Changes Using News with Gemini
In the realm of finance, artificial intelligence (AI) and machine learning (ML) are revolutionizing
the way we analyze markets. Gemini, a powerful large language model (LLM) from Google AI, offers exceptional natural language processing (NLP) capabilities, paving the way for innovative
methods in stock price prediction. This blog post delves into how Gemini can be employed to analyze economic and financial news and derive insights into potential stock price movements.
Understanding Gemini
Gemini is a state-of-the-art LLM developed by Google AI, adept at generating human-quality text
based on the vast datasets it has been trained on. While its primary focus lies in text generation and
translation, Gemini's remarkable ability to identify and learn from intricate patterns makes it a valuable asset for financial market analysis.
Why Utilize Gemini for Stock Price Prediction?
1. Unveiling Hidden Patterns: Gemini excels at recognizing complex relationships within
data, a crucial skill for anticipating stock prices, which are influenced by a multitude of factors.
2. Natural Language Expertise: Gemini excels at processing and analyzing unstructured data
from financial news sources, social media, and market reports, contributing to a more comprehensive prediction model.
3. Adaptable to Change: Gemini's flexibility allows for fine-tuning with specific datasets, enabling it to adapt to diverse market conditions and sectors.
Harnessing Gemini for Stock Price Prediction
1. Data Gathering
To train Gemini for stock price prediction, a rich tapestry of data sources is essential:
• Historical Stock Prices: Time-series data encompassing stock prices and trading volumes.
• Financial News Articles and Reports: Reputable sources providing valuable insights.
• Social Media Sentiment Analysis: Quantitative assessment of sentiment (positive, negative, neutral) expressed on social media platforms regarding the stock or market.
• Economic Indicators: Data such as interest rates, inflation rates, and GDP growth.
2. Data Preparation
Before feeding the data into Gemini, it's crucial to clean and structure it effectively:
• Historical Data: Normalize and scale stock prices to ensure appropriate model training.
• News and Social Media: Utilize sentiment analysis to convert textual data into numerical
values representing sentiment (positive, negative, neutral).
3. Model Training
The preprocessed data is fed into Gemini, embarking on the training journey. Here's a simplified
representation of the process:
1. Tokenization: Text data is segmented into smaller units (tokens) that Gemini can comprehend.
2. Training: Gemini undergoes refinement to anticipate the next token (word) in a sequence.
This capability is then adapted to predict the next stock price.
3. Fine-Tuning: The model's parameters are meticulously adjusted to minimize prediction errors and enhance accuracy.
4. Prediction and Assessment
Once trained, Gemini can forecast future stock prices. Employing evaluation metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) is crucial to assess the reliability of
these predictions. Continuous training with new data ensures the model stays updated with evolving
market conditions.
Benefits of Leveraging Gemini for Stock Price Prediction
• Comprehensive Analysis: Integrates diverse data sources for more accurate predictions.
• Real-Time Processing: Swiftly analyzes and responds to new information.
• Customization: Adaptable to specific markets, industries, or stocks.
Challenges and Considerations
• Data Quality: The accuracy of predictions hinges on high-quality input data. Incomplete or
biased data can lead to misleading forecasts.
• Market Volatility: Financial markets are inherently unpredictable, and unexpected events
can disrupt underlying patterns.
• Computational Resources: Training large LLMs like Gemini requires substantial computational power and memory.
Getting Started with Gemini for Stock Prediction
While a comprehensive guide is beyond the scope of this post, here are the essential tools and libraries to get you started:
Tools and Libraries
• Python: The primary programming language for AI and ML projects.
• TensorFlow or PyTorch: Deep learning frameworks for model training and fine-tuning.
• Pandas and NumPy: Libraries for data manipulation and analysis.
• Scikit-learn: Provides tools for data preprocessing and evaluation metrics.
• Google Cloud Platform (GCP): Offers access to Gemini and the necessary computational
resources (optional).
Basic Implementation Steps
Importing necessary libraries:
import feedparser
import google.generativeai as genai
import re
import json
Download the economic news:
rss_url = 'https://www.ekonomidunya.com/rss_ekonomi_1.xml'
feed_eco = feedparser.parse(rss_url)
for entry in feed_eco.entries:
print(f"Published: {entry.published}")
print(f"Title: {entry.title}")
print(f"Summary: {entry.summary}")
print(f"Content: {entry.content[0].value}")
print("-" * 80)
Adding news to a list:
def feed_to_list(feed, list):
for entry in feed.entries:
entry_str = f"Published: {entry.published} \n" + f"Title: {entry.title} \n"
+ f"Summary: {entry.summary} \n" + f"Content: {entry.content[0].value}"
list.append(entry_str)
return list
eco_list = []
eco_list = feed_to_list(feed_eco, eco_list)
eco_list = list(reversed(eco_list))
Loading current stock data and reading last day stock scores:
import pandas as pd
df = pd.read_csv('stock_data_daily.csv')
df.Date = pd.to_datetime(df.Date)
df = df.set_index('Date')
df.tail()
last_day_row = df.tail(1)
last_day_row_index = df.tail(1).index
score_columns = [col for col in df.columns if col.startswith('score_')]
last_day = last_day_row[score_columns]
keys = last_day.columns.to_list()
values = last_day.values.reshape((len(last_day.columns.to_list()), ))
result_dict = dict(zip(keys, values))
result_dict
System instruction:
system_str = "As a Market Analyst utilizing the OpenAI API key, your role involves
analyzing daily economic news and its potential impact on specific stocks listed on
Borsa Istanbul (BIST). \
You are tasked with evaluating the relevance of news articles to the current
scores of given stocks and adjusting their scores accordingly. Here's how your role
operates: \
Daily Task Overview: \
Read and analyze daily economic news articles. \
Identify any information that could potentially influence the prices of
specified stocks listed on Borsa Istanbul. \
Stock Score System: \
Each stock has an initial score of 50.0, representing a neutral stance. \
Your analysis will adjust these scores based on the perceived impact of the
news on the respective stocks. \
If significant information is found that could affect a stock's price, you
will update its score accordingly on a scale of 0 to 100. \
Adjusting Stock Scores: \
If the news suggests a potential impact on a stock's price, assess the degree of influence and adjust the stock's score accordingly. \
Scores can be adjusted upwards or downwards based on the perceived impact.
\
No Impact Scenario: \
If the analysis does not reveal any information likely to affect a stock's
price, maintain the stock's score at its current level. \
Scoring Criteria: \
Consider various factors such as financial performance, market sentiment,
industry trends, and any other relevant data when adjusting scores. \
Use a holistic approach to determine the magnitude of score adjustments,
ensuring accuracy and consistency. \
Documentation and Reporting: \
Document your analysis process, including key findings and the rationale
behind score adjustments. \
Provide regular reports summarizing the impact of news articles on stock
scores and any notable market trends. \
By effectively analyzing economic news and adjusting stock scores accordingly, you
play a crucial role in providing valuable insights for investors and stakeholders
in the Borsa Istanbul market."
Configuring Gemini model (before the user must create a new Gemini API key):
API_KEY = 'YOUR_GEMINI_API_KEY'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest',
system_instruction=system_str)
Model summary:
model
Function for splitting score dictionary and analysis article:
def str_to_score_dict(old_dict, res_text):
pattern = r'\{.*\}'
match = re.search(pattern, res_text)
if match:
dict_string = match.group()
result_dict = json.loads(dict_string)
remaining_string = res_text.replace(dict_string, '')
return result_dict, remaining_string
else:
return old_dict, ""
Function for updating stock scores:
def score(eco_list, result_dict):
analyze_list = []
list_length = len(eco_list)
for i in range(0, list_length, 5):
news = ""
chunk = eco_list[i:i+5]
for new in chunk:
news = news + new + "\n\n"
str_dict = json.dumps(result_dict)
response = model.generate_content(str_dict + '\n' + news)
result_dict, analyze = str_to_score_dict(result_dict, response.text)
analyze_list.append(analyze)
return result_dict, analyze_list
Calling function and updating scores:
new_result_dict, analyze_document = score(eco_list, result_dict)
Printing analysis:
for analysis in analyze_document:
print(analysis)
Updating last row:
keys1 = last_day_row.columns.to_list()
values1 = last_day_row.values.reshape((len(last_day_row.columns.to_list()), ))
dict_final = dict(zip(keys1, values1))
dict_final
def replace_dict_values(dict1, dict2):
updated_dict = {key: dict2.get(key, value) for key, value in dict1.items()}
return updated_dict
dict_final = replace_dict_values(dict_final, new_result_dict)
dict_final
for key, value in dict_final.items():
df.loc[last_day_row_index, key] = value
Saving new scores:
df.to_csv('stock_data_daily.csv')
Conclusion
Gemini ability to process and analyze large datasets, coupled with its pattern recognition
capabilities, makes it a powerful tool for predicting stock price changes. By leveraging financial
news, social media sentiment, and historical data, Gemini can provide more accurate and timely
predictions. As AI technology evolves, we can expect even more sophisticated and reliable
prediction models, ushering in a new era of AI-driven finance.
