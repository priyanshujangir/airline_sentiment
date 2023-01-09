
from fastapi import Depends, FastAPI
import pickle
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel
import re


app = FastAPI()


class text(BaseModel):
    tweet: str


def text_cleaner(tweet):
    tweet = re.sub('@[\w]+', '', tweet)  # removes username handles
    tweet = re.sub(r"http\S+", "", tweet)  # removes links/urls
    tweet = re.sub(r'#', '', tweet)  # removes "#"
    # removes repeating characters and replaces with single character
    tweet = re.sub(r'([A-Za-z])\1{2,}', r'\1', tweet)

    # only number allowed is zero in alphabet form, all other omitted
    tweet = re.sub(r' 0 ', 'zero', tweet)
    tweet = re.sub(r'[^A-Za-z ]', '', tweet)

    tweet = tweet.lower()
    return tweet


@app.get('/')
def index():
    return {'message': 'Hello, world'}


@app.post('/predict')
async def predict_sentiment(data: text):
    loaded_model = pickle.load(open('randomf_model.pkl', 'rb'))
    vector = pickle.load(open('vectors.pkl', 'rb'))
    text = data.tweet
    text = text_cleaner(text)
    text = [text]
    vec = vector.transform(text)
    prediction = loaded_model.predict(vec)
    prediction = int(prediction)
    if (prediction > 0):
        prediction = "positive"
    else:
        prediction = "negative"

    return {"sentence": data.tweet, "prediction": prediction}


if __name__ == '__main__':
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, host='127.0.0.1', port=8000)
