import os
import pandas as pd
import requests
import shutil
from tqdm import tqdm

post = pd.read_pickle("./post.pkl")

cwd = os.getcwd()
if not os.path.exists(cwd+'/data/img/'):
    os.makedirs(cwd+'/data/img/')

urls = list(post.img_urls.str.strip('{}'))
n =1
print("downloading imgs with urllib")
for url in tqdm(urls):
    if ',' in url:
        url = url.split(',')[0]
    r = requests.get(url, stream=True)
    with open(cwd+"/data/img/"+str(n)+".jpg", "wb") as img:
        shutil.copyfileobj(r.raw, img)
    n += 1
    del r