
from flask import Flask, request, jsonify
from tweet_generator import SimpleTweetGenerator
import joblib
from textblob import TextBlob
import numpy as np
import pandas as pd

app = Flask(__name__)
generator = SimpleTweetGenerator()
    
label_encoder = joblib.load("label_encoder.joblib")
model = joblib.load("like_predictor.pkl")


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        
        
        company = data.get('company', 'Our Company')
        tweet_type = data.get('tweet_type', 'general')
        message = data.get('message', 'Something awesome!')
        topic = data.get('topic', 'innovation')
        
        
        generated_tweet = generator.generate_tweet(company, tweet_type, message, topic)
        
        return jsonify({
            'generated_tweet': generated_tweet,
            'success': True,
            'company': company,
            'type': tweet_type
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Tweet Generator API is running!'})

@app.route('/generate_and_predict', methods=['POST'])
def generate_and_predict():
    """Generate a tweet AND predict how many likes it will get!"""
    data = request.get_json()

    company = data.get('company', 'Our Company')
    tweet_type = data.get('tweet_type', 'general')
    message = data.get('message', 'Something awesome!')
    topic = data.get('topic', 'innovation')
        
        
    generated_tweet = generator.generate_tweet(company, tweet_type, message, topic)

    def extract_features_from_tweet(gen_tweet,company):
        char_count = len(gen_tweet)
        word_count = len(gen_tweet.split())
        company_encoded = label_encoder.transform([company])[0]
        sentiment = TextBlob(gen_tweet).sentiment.polarity

        return word_count, char_count, sentiment, company_encoded


    features = extract_features_from_tweet(generated_tweet ,company)

    columns = ["word_count", "char_count", "sentiment", "company_encoded"]
    features_df = pd.DataFrame([features], columns=columns)

    predicted_log_likes = model.predict(features_df)[0]

    predicted_likes = np.expm1(predicted_log_likes)

    
    return jsonify({
        'generated_tweet': generated_tweet,
        'predicted_likes': int(predicted_likes),
        'success': True
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)  