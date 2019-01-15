from bs4 import BeautifulSoup
import requests
import re, nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')


group_names = ['Nightwing32193', 'RealZachKraus', 'Graehamkracker', 'BeefyChickens', 'ButcherDawg', "Dragodar"]
character_names = ['Gideon', 'Chadwick', 'Bendron', 'Demryon', 'Seamus', 'GM']
group_colors = ["#FF0000","#0000FF","#8E5500","#800080","#008000", "#000000"]
url = "http://www.giantitp.com/forums/showthread.php?553965-IC-The-Storm-on-the-Horizon"

def glean_rolls(post):

    name_grep = "|".join(group_names)
    author = re.search(name_grep, str(post.find_all(class_="siteicon_profile"))).group()
    
    roll_block = re.findall("\(.d.*\)\[<b>.*<\/b>\]", str(post.find_all(class_="postbody")))
    if(len(roll_block) > 0):
        out_data = pd.DataFrame()
        for roll in roll_block:
            roll_nums = re.findall(r'\d+', roll)
            num_dice = int(roll_nums[0])
            dice_type = int(roll_nums[1])
            modifier = int(roll_nums[2])
            score = int(re.findall(r'\d+', re.findall("\>.*\<", roll)[0])[0])
            if modifier == score:
                modifier = 0

            rolled = score - modifier
            normalized = rolled/(num_dice * dice_type)
            out_line = {"Author":author, "Rolled":rolled, "Normalized":normalized}
            out_line = pd.DataFrame(out_line, index=[0])
            out_data = pd.concat([out_data, out_line], ignore_index=True)

    else:
        out_data = {"Author":author, "Rolled":None, "Normalized":None}
        out_data = pd.DataFrame(out_data, index=[0])

    return(out_data)


def glean_text(post):

    text_remove = '<blockquote class="postcontent restore ">|<font color=".*">|<b>|<...........>|<div class="spoiler">|Timestamp|</.>|</....>|</...>|<.*">|<br/>|<img alt="" border="0" src=".*"/>|<div align="left" class="spoiler-title">|\t|\n|\r|\[|\]|Show|Youtube'
    name_grep = "|".join(group_names)
    author = re.search(name_grep, str(post.find_all(class_="siteicon_profile"))).group()
    #author = re.search(name_grep, str(post.find_all(class_="postcontent lastedited"))).group()
    text = re.sub(text_remove, "",  str(post.find_all(class_="postcontent restore ")))
    out_data = {"Author":author, "Text":text}
    out_data = pd.DataFrame(out_data, index=[0])

    return(out_data)


session = requests.session()
response = session.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# locate the last page link
last_page_link = soup.find(class_="first_last")
last_page_number = re.search(r'\d+', re.search(r"page(\d+)&", str(last_page_link)).group()).group()


roll_data = pd.DataFrame()
text_data = pd.DataFrame()
for page_num in range(int(last_page_number)+1):
    page = "http://www.giantitp.com/forums/showthread.php?553965-IC-The-Storm-on-the-Horizon/page"+str(page_num)
    r = requests.get(page)
    soup = BeautifulSoup(r.content)

    #soup.prettify()
    #postbody
    #posthead
    #postdetails
    posts = soup.find_all(class_="postdetails")
    #soup.find_all(class_="postdetails")
    for post in posts:
        roll_data = roll_data.append(glean_rolls(post), ignore_index=True)
        text_data = text_data.append(glean_text(post), ignore_index=True)

text_data.to_csv("Post_Text.csv")

roll_data.groupby("Author").Normalized.mean()
roll_data.groupby("Author").Normalized.std()
#roll_data.groupby("Author").count()

nltk.download('punkt')
ps = nltk.stem.snowball.SnowballStemmer('english')

from sklearn.base import BaseEstimator, TransformerMixin
class to_corpus(BaseEstimator, TransformerMixin):
    def __init__(self, remove):
        self.remove = remove

    def transform(self, X, *_):
        #X = [X]
        cleaned = [re.sub(self.remove, " ", text) for text in X]
        result = [" ".join([w for w in line.lower().split(" ") if not w in stop_words]) for line in cleaned]

        #result = [" ".join([ps.stem(w) for w in line.lower().split(" ") if not w in stop_words]) for line in cleaned]
        return result

    def fit(self, *_):
        return self

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(["1d20"])

'''from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('to_corpus', to_corpus("[^A-z]")),
    ('transformer', CountVectorizer(ngram_range=(1,3)))
])

matrix = pipeline.fit_transform(text_data.Text.astype(str).tolist())
sum_words = matrix.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in pipeline.named_steps.transformer.vocabulary_.items()]
sorted(words_freq, key = lambda x: x[1], reverse=True)[0:10]'''


from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from os import path

text = " ".join(to_corpus("[^A-z]").transform(text_data.Text.astype(str).tolist()))
#text = " ".join(to_corpus("[^A-z]").transform(text_data[text_data.Author == "Dragodar"].Text.astype(str).tolist()))
wordcloud = WordCloud().generate(text)

'''plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()'''

mask = np.array(Image.open("Images/Seamus.jpg"))
image_colors = ImageColorGenerator(mask)
#image_colors = ImageColorGenerator(mask)
wordcloud = WordCloud(background_color="white", mask=mask, max_words=200000, color_func=image_colors, max_font_size=90).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('Outputs/Seamus.png', format='png', dpi=2000)
plt.show()