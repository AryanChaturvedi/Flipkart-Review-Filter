import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import nltk
import contractions
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import re, string, unicodedata
from keybert import KeyBERT
import nltk
import contractions
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import re, string, unicodedata

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from textblob import TextBlob

# CSS to inject contained in a string
hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.write(
    """
                <style type="text/css" media="screen">
                div[role="listbox"] ul {
                    height:300px;
                }
                </style>
                """
    ,
    unsafe_allow_html=True,
)
st.title('Flipkart Feature Based Review Extractor & Question Answering')
st.markdown("""
            ## What Can You do with this project 
            - you can filter any reviews based on specific feature of product. for example: if product is a mobile then you can filter reviews talking about camera or battery or any other specifications.
            - Instead of reading 1000 of reviews and getting info about particular feature, you can pass all reviews to this project and get a one line sentence(talking about searched review) extracted from each reviews metioning that feature.
            - you can get a user sentiment review score for individual feature
            - You can ask any querry and if it is somewhere mentioned in any review then our model will show closest matching sentence to related querry. 
            """)

from PIL import Image
image = Image.open('Inkedflipkart.jpg')
st.image(image, caption='Click on highlighted all review link')
image2 = Image.open('flipkart2.png')
st.image(image2,'Copy above mentioned link')
st.subheader("Enter Product's All review Page copied link")
base = st.text_input('Enter Review link')

st.markdown("""
             Model is scrapping 400 review pages to collect 1000+ User reviews.
            - Web scrapping of 400+ pages takes time (approx 3-5min).
        """)
@st.cache(allow_output_mutation=True)
def scraper(base):
    r = requests.get(base)
    html = r.content
    soup = BeautifulSoup(html,  "html.parser")
    divs_r = soup.find_all("div",{"class":"col _2wzgFH K0kLPL"})
    data_e = pd.DataFrame({
        'Rating':pd.Series(dtype='int'),
        'Title': pd.Series(dtype='str'),
        'Review':pd.Series(dtype='str')
    })

    # divs_p = soup.find_all("div",{"class": "_2MImiq _1Qnn1K"})
    # pageno = divs_p[0].find('span').text
    # pages = [int(i) for i in pageno.split() if i.isdigit()]
    # total_pages= pages[1]
    for i in range(400):
        base = base + '&page='+str(i)
        r = requests.get(base)
        html = r.content
        soup = BeautifulSoup(html,  "html.parser")
        divs_r = soup.find_all("div",{"class":"col _2wzgFH K0kLPL"})
        ratings = [int(soup.new_string("3")) if divs_r.find('div',class_='_3LWZlK _1BLPMq') is None else int(divs_r.find('div',class_='_3LWZlK _1BLPMq').text.strip('')) for divs_r in divs_r]
        titles = [soup.new_string("") if divs_r.find('p',class_='_2-N8zT') is None else divs_r.find('p',class_='_2-N8zT').text.strip('') for divs_r in divs_r]
        review = [soup.new_string("") if divs_r.find_all('div',class_='t-ZTKy') is None else divs_r.find_all('div',class_='t-ZTKy')[0].get_text(separator=" ").strip() for divs_r in divs_r]
        data = pd.DataFrame({
        'Rating':ratings,
        'Title': titles,
        'Review':review})
        data_e=data_e.append(data,ignore_index=True)
        if data_e.shape[0] > 1000:
            break
    df = data_e
    for i in range(df.shape[0]):
        df['Review'][i] = df['Review'][i][:-9]

    return df
df2 = scraper(base)
st.write(scraper(base))

## Rating Graph
temp = df2
temp['count']=1
temp = temp.groupby('Rating')['count'].sum()
temp = pd.DataFrame(temp)
temp = temp.reset_index()
total = temp['count'].sum()
temp['percent']= (temp['count']/total)*100
temp = temp.round(2)
import plotly.graph_objects as go

fig = go.Figure(go.Bar(
            x=temp['count'],
            y= temp['Rating'],
            text = temp['percent'],
            orientation='h'))
st.subheader( 'Overall User rating')
st.plotly_chart(fig)
## Topic extraction
@st.cache(allow_output_mutation=True)
def Topic_extraction(df2):
    model = KeyBERT(model="distilbert-base-nli-mean-tokens")
    mobile = df2
    # function for extracting keyword
    def Extract(lst):
        return list(item[0] for item in lst)

    # extracting keywords from each review
    keywords2 = []
    for i in range(len(mobile)):
        key = model.extract_keywords(
            mobile['Review'][i],
            top_n=15,
            keyphrase_ngram_range=(1, 1),  ## (1,1) unigrame keywords
            stop_words="english",
        )
        keywords2.append(Extract(key))

    mobile['keywords2'] = keywords2

    # creating a vocabulory from kewwords of all review
    lst2 = mobile['keywords2'].explode().to_list()
    return lst2,mobile
# Creating a frequency distribution of kewwords
lst2,mobile2 =  Topic_extraction(df2)
from nltk import FreqDist
fdist2 = FreqDist(lst2)
freq2 = pd.DataFrame(fdist2.items(), columns=['word', 'frequency'])
freq2 = freq2.sort_values('frequency', ascending=False).reset_index()
freq2 = freq2.drop('index', axis=1)
freq2 = freq2[3:]

# Creating a wordcloud
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS


text = ' '.join(mobile2['Review'])
stopwords  = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=42, min_font_size=8, background_color="white").generate(
    text)
fig, ax = plt.subplots(figsize = (15, 10))
ax.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

st.markdown("""
    #### Word cloud of most frequent words in all reviews.
    - You can pick any specifications mentioned in word cloud or by yourself and filter reviews or ask a querry related to selected specification.
""")
st.pyplot(fig)

st.subheader('Select Review Search or Question Answer')
st.markdown("""
            - Select Review Filter if you want to filter Revies and get one line review for particular specification.
            - Select Ask Question if you want to querry about a specification.
            """)
fxn = st.selectbox('Select Querry',['Select','Review Filter','Ask a Question'])
if (fxn== 'Review Filter'):
    ## MAKING AN ASPECT BASED REVIEW EXTRACTOR
    new_row = pd.DataFrame({'word':'Select Feature', 'frequency':1}, index=[0])
    freq2 = pd.concat([new_row,freq2.loc[:]]).reset_index(drop=True)
    selected = st.selectbox('Select Specification',freq2['word'])

    st.markdown("""
                - Our model is extracting all sentences related to our specification from 1000+ reviews.
                - Short Review column contains summary of review for that specification only.
                - It saves a lot of time reading for whole big revies to get an idea about a particular specification.
                - It is a deep learning based model so it may take some time for excecution.  
                """)

    # Creating an aditional review column with all lowercase words
    mobile2['Review_c'] = mobile2['Review'].str.lower()
    # Filtering reviews based on searched aspect
    df_filt = mobile2[mobile2['Review_c'].str.contains(selected)]
    df_filt = df_filt.reset_index()
    df_filt = df_filt.drop('index', axis=1)

    # Remove emojis
    import re
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def html_remover(data):
            beauti = BeautifulSoup(data, 'html.parser')
            return beauti.get_text()

    def url_remover(data):
            return re.sub(r'https\S', '', data)

    def web_associated(data):
            text = html_remover(data)
            text = url_remover(text)
            return text

    def remove_round_brackets(data):
            return re.sub('\(.*?\)', '', data)

    def remove_punc(data):
            trans = str.maketrans('', '', string.punctuation)
            return data.translate(trans)

    def white_space(data):
            return ' '.join(data.split())

    def complete_noise(data):
            new_data = url_remover(data)
            new_data = web_associated(new_data)
            new_data = remove_round_brackets(new_data)

            new_data = white_space(new_data)
            return new_data

    df_filt['Review_c'] = df_filt['Review_c'].apply(complete_noise)

    ## Applying BERT question answering model

    # Model
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    @st.cache(allow_output_mutation=True)
    def answer_question(question, answer_text):
            '''
            Takes a `question` string and an `answer_text` string (which contains the
            answer), and identifies the words within the `answer_text` that are the
            answer. Prints them out.
            '''
            # ======== Tokenize ========
            # Apply the tokenizer to the input text, treating them as a text-pair.
            input_ids = tokenizer.encode(question, answer_text)

            # ======== Set Segment IDs ========
            # Search the input_ids for the first instance of the `[SEP]` token.
            sep_index = input_ids.index(tokenizer.sep_token_id)

            # The number of segment A tokens includes the [SEP] token istelf.
            num_seg_a = sep_index + 1

            # The remainder are segment B.
            num_seg_b = len(input_ids) - num_seg_a

            # Construct the list of 0s and 1s.
            segment_ids = [0] * num_seg_a + [1] * num_seg_b

            # There should be a segment_id for every input token.
            assert len(segment_ids) == len(input_ids)

            # ======== Evaluate ========
            # Run our example through the model.
            outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                            token_type_ids=torch.tensor([segment_ids]),
                            # The segment IDs to differentiate question from answer_text
                            return_dict=True)

            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            # ======== Reconstruct Answer ========
            # Find the tokens with the highest `start` and `end` scores.
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)

            # Get the string versions of the input tokens.
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Start with the first token.
            answer = tokens[answer_start]

            # Select the remaining answer tokens and join them with whitespace.
            for i in range(answer_start + 1, answer_end + 1):

                # If it's a subword token, then recombine it with the previous token.
                if tokens[i][0:2] == '##':
                    answer += tokens[i][2:]

                # Otherwise, add a space then the token.
                else:
                    answer += ' ' + tokens[i]

            # print('Answer: "' + answer + '"')
            return answer

    querry = []
    question = "how is " + selected

    for i in range(len(df_filt)):
            bert_abstract = df_filt['Review_c'][i]
            querry.append(answer_question(question, bert_abstract))
    querry = pd.DataFrame(querry, columns=['Short Review'])
    querry['Full Review'] = mobile2['Review']
        ## show querry

    st.write(querry)
        ## Applying sentiment score using textblob
    @st.cache(allow_output_mutation=True)
    def sentiment_analysis(querry):
        querry['sentiment'] = querry['Short Review'].apply(lambda x: (TextBlob(x).sentiment.polarity))
        positive= 0
        negative = 0
        neutral = 0
        for i in range(len(querry)):
            if querry['sentiment'][i]>0:
                positive= positive + 1
            elif querry['sentiment'][i]<0:
                negative= negative + 1
            else:
                 neutral= neutral + 1

        return positive,negative,neutral


    positive, negative, neutral= sentiment_analysis(querry)
    sum = positive + negative + neutral
    fig = go.Figure(go.Bar(
                y=[positive,negative,neutral],
                x=['Positive','Negative','Neutral'],
                text = [(round((positive/sum)*100),2),round(((negative/sum)*100),2),round(((neutral/sum)*100),2)]
                ))
    st.markdown("""
                - From extracted summary review about particular specification we now apply sentiment analysis to get average score for particular specification.
                - Below is positive vs negative user review about searched specification 
                """)
    st.plotly_chart(fig)

if (fxn == 'Ask a Question'):
    st.markdown("""
                - Ask any type of querry and our model will extract sentences related to mentioned querry.
                - if there are no reviews or sentences related to querry then model will return empty or [CLS] querry [SEP] type sentences.
                - for Example: if selected product is a smartphone then querry = how is battery performance.
                - It is very basic model so can give absurd answers too.
                """)
    question = st.text_input('Enter Question')
    corpus = ' '.join(mobile2['Review'])

    ## Applying BERT question answering model

    # Model
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


    @st.cache(allow_output_mutation=True)
    def answer_question(question, answer_text):
        '''
        Takes a `question` string and an `answer_text` string (which contains the
        answer), and identifies the words within the `answer_text` that are the
        answer. Prints them out.
        '''
        # ======== Tokenize ========
        # Apply the tokenizer to the input text, treating them as a text-pair.
        input_ids = tokenizer.encode(question, answer_text)

        # ======== Set Segment IDs ========
        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
        segment_ids = [0] * num_seg_a + [1] * num_seg_b

        # There should be a segment_id for every input token.
        assert len(segment_ids) == len(input_ids)

        # ======== Evaluate ========
        # Run our example through the model.
        outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                        token_type_ids=torch.tensor([segment_ids]),
                        # The segment IDs to differentiate question from answer_text
                        return_dict=True)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # ======== Reconstruct Answer ========
        # Find the tokens with the highest `start` and `end` scores.
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # Get the string versions of the input tokens.
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Start with the first token.
        answer = tokens[answer_start]

        # Select the remaining answer tokens and join them with whitespace.
        for i in range(answer_start + 1, answer_end + 1):

            # If it's a subword token, then recombine it with the previous token.
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]

            # Otherwise, add a space then the token.
            else:
                answer += ' ' + tokens[i]

        # print('Answer: "' + answer + '"')
        return answer

    start = 0
    end = 1000
    while end < (len(corpus) / 4):
        st.write(answer_question(question, corpus[start:end]))
        start = start + 1000
        end = end + 1000

