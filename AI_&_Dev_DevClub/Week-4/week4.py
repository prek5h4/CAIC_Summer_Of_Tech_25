
from flask import Flask, request, jsonify , redirect, url_for,render_template, request
from tweet_generator import SimpleTweetGenerator
import joblib
from textblob import TextBlob
import numpy as np
import pandas as pd

app = Flask(__name__)
generator = SimpleTweetGenerator()
    
label_encoder = joblib.load("label_encoder.joblib")
model = joblib.load("like_predictor.pkl")

@app.route("/")
def home():
    return render_template("index.html")

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


@app.route('/generate_and_predict', methods=['POST', 'GET'])
def generate_and_predict():

        if request.is_json:
            data = request.get_json()
            company = data.get('company', 'Our Company')
            tweet_type = data.get('tweet_type', 'general')
            message = data.get('message', 'Something awesome!')
            topic = data.get('topic', 'innovation')
        else:
        
            company = request.form.get('nm', 'Our Company')  
            tweet_type = request.form.get('tt', 'general')   
            message = request.form.get('msg', 'Something awesome!')  
            topic = request.form.get('tp', 'innovation')    
    
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

@app.route("/form", methods=["POST","GET"])
def form():
        if request.method == "POST":
            
            company = request.form.get("nm", "Our Company")
            tweet_type = request.form.get("tt", "general")
            message = request.form.get("msg", "Something awesome!")
            topic = request.form.get("tp", "innovation")
            
           
            generated_tweet = generator.generate_tweet(company, tweet_type, message, topic)

            def extract_features_from_tweet(gen_tweet, company):
                char_count = len(gen_tweet)
                word_count = len(gen_tweet.split())
                company_encoded = label_encoder.transform([company])[0]
                sentiment = TextBlob(gen_tweet).sentiment.polarity
                return word_count, char_count, sentiment, company_encoded

            features = extract_features_from_tweet(generated_tweet, company)
            columns = ["word_count", "char_count", "sentiment", "company_encoded"]
            features_df = pd.DataFrame([features], columns=columns)
            predicted_log_likes = model.predict(features_df)[0]
            predicted_likes = np.expm1(predicted_log_likes)

            return jsonify({
                'generated_tweet': generated_tweet,
                'predicted_likes': int(predicted_likes),
                'success': True,
                'form_data': {
                    'company': company,
                    'tweet_type': tweet_type,
                    'message': message,
                    'topic': topic
                }
            })
        else:
            return render_template("form.html")
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)  