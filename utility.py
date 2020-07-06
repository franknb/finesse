import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk; nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
# Utility function that returns compound vader sentiment score for given text
def sentiment(text):
    return analyser.polarity_scores(text)['compound']

import emoji
import regex
# Utility function that removes emoji in string
def remoji(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c not in emoji.UNICODE_EMOJI]
    return ''.join(emoji_list)

# Utility function that counts emoji in string
def countemoji(text):
    data = regex.findall(r'\X', text)
    emojis = [char for char in data if char in emoji.UNICODE_EMOJI]
    return len(emojis)

# Utility function for remove |, @ and # from comments
def process_comment(comment):
    text = remoji(comment).replace('|', ' ').split()
    text_list = [x for x in text if not x.startswith(('@','#'))]
    return " ".join(text_list)

from googletrans import Translator
from tqdm.auto import tqdm; tqdm.pandas()
# utility function for translate all comments into English
def trans(post):
    for i in tqdm(range(len(post))):
        if i%400 == 0:
            t = Translator()
        try:
            translations = t.translate(post.loc[i, 'en_comment'], dest='en')
            post.loc[i, 'en_comment'] = translations.text.replace('\u200b', '').replace('\u200d', '')
        except:
            pass
        
import datetime
# utility function to generate weekday from date string
def getweekday(datestring):
    d = datetime.datetime.strptime(datestring, '%Y-%m-%d')
    return d.weekday()

# utility functions to generate weekday from date string
def getdate1(datestring):
    d = datetime.datetime.strptime(datestring, '%Y-%m-%d')
    return d
def getdate2(datestring):
    d = datetime.datetime.strptime(datestring, '%Y-%m-%d %H:%M:%S')
    return d

def findindex(df, xticks):
    index = []
    for i in xticks:
        index.append(df.iloc[(df.username-i).abs().argsort()[:1]].index[0])
    return list(set(index))

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
# A utility function for plotting ROC curves
def plot_curves(label, preds, C = True, lab = ''):
    fpr, tpr, _ = roc_curve(label, preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label, preds)
    f1 = f1_score(label, preds>0.5)
    plt.subplot(1,2,1)
    if C == True:
        plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = lab + 
                 'ROC curve (area = {0:0.3f})'.format(roc_auc))
    else:
        plt.plot(fpr, tpr, lw = 2, label = lab + 'ROC curve (area = {0:0.3f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle='--')
    plt.xlim([-0.01, 1.0]);plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.subplot(1,2,2)
    if C == True:
        plt.step(recall, precision, color='b', alpha=0.2, where='post', label = lab + 
                 'Precision-Recall curve: F1={0:0.3f}'.format(f1))
        plt.fill_between(recall, precision, alpha=0.2, color='b', step = 'post')
    else:
        plt.step(recall, precision, where='post', label = lab + 'Precision-Recall curve: F1={0:0.3f}'.format(f1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.0])
    plt.legend()