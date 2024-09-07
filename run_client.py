# client.py
import requests
import numpy as np
from flask import jsonify

# 서버의 URL
if __name__=='__main__':
    url = 'http://172.27.190.155:100'

    image_list = np.ones([5, 256, 256, 3])
    ft_list = np.ones([5, 6])
    proprio_list = np.ones([5, 7])
    response = requests.post(url + '/predict', json={'images': image_list.tolist(), 'ft': ft_list.tolist(), 'proprio': proprio_list.tolist()})
    response = response.json()
    action = list(map(float, response['action']))