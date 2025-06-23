import requests

response = requests.post('http://localhost:5001/generate_and_predict', json={
    'company': 'starbucks',
    'tweet_type': 'question',
    'message': 'trying new recipes',
    'topic': 'coffee'
})

print("Status Code:", response.status_code)
print("Raw Response Text:", response.text) 
try:
    print("JSON Response:", response.json())
except Exception as e:
    print("JSON decode error:", e)
