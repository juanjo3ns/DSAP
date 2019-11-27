import requests
import json
import os

token = os.environ['TG_KEY']
url = 'https://api.telegram.org/bot' + token+ '/sendMessage'

def send(message):
    try:
        r = requests.post(
            url=url,
            data={'chat_id': -377175234, 'text': message}
        ).json()
    except:
        pass
