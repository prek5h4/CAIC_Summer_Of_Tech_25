import joblib
import numpy as np
from textblob import TextBlob
import pandas as pd


label_encoder = joblib.load("label_encoder.joblib")
model = joblib.load("like_predictor.pkl")

def extract_features_from_tweet(gen_tweet, company):
    
    word_count = len(gen_tweet.split())

    char_count = len(gen_tweet)

    sentiment = TextBlob(gen_tweet).sentiment.polarity

    company_encoded = label_encoder.transform([company])[0]

    return [word_count, char_count, sentiment, company_encoded]

# Example input
generated_tweet = "Hey everyone! we have launched refreshing coffee"
company = "starbucks"
features = extract_features_from_tweet(generated_tweet ,company)
# Extract features
columns = ["word_count", "char_count", "sentiment", "company_encoded"]
features_df = pd.DataFrame([features], columns=columns)

predicted_log_likes = model.predict(features_df)[0]

predicted_likes = np.expm1(predicted_log_likes)

print("Predicted likes:", int(predicted_likes))
