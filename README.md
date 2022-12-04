# Flipkart-Review-Filter

## What Can You do with this project

- you can filter any reviews based on specific feature of product. for example: if product is a mobile then you can filter reviews talking about camera or battery or any other specifications.
- Instead of reading 1000 of reviews and getting info about particular feature, you can pass all reviews to this project and get a one line sentence(talking about searched review) extracted from each reviews metioning that feature.
- you can get a user sentiment review score for individual feature
- You can ask any querry and if it is somewhere mentioned in any review then our model will show closest matching sentence to related querry.

![Inkedflipkart](https://user-images.githubusercontent.com/77160352/205491184-62ed7416-8cc9-4cb3-973b-242ff3c74624.jpg)

Click on highlighted all review link

![flipkart2](https://user-images.githubusercontent.com/77160352/205491226-2134ef43-200a-43bd-9e96-3fe3e603c1e3.png)

- Model will scrap 400 review pages to collect 1000+ User reviews.
- Web scrapping of 400+ pages takes time (approx 3-5min).
  
  ![image](https://user-images.githubusercontent.com/77160352/205491682-1545769f-0843-4d4e-a372-4fd53f1600d4.png)

  ![image](https://user-images.githubusercontent.com/77160352/205491641-7b5d9c60-be7d-44ec-a7d6-fbc2091e0963.png)
  
  
Based on most frequent specification or word mentioned in reviews, our model will extract keywords using KeyBERT

- Word cloud of most frequent words in all reviews is generated.
![image](https://user-images.githubusercontent.com/77160352/205491747-4529554c-5031-44ca-8ec5-b094d413dd05.png)

- You can pick any specifications mentioned in word cloud or by yourself and filter reviews or ask a querry related to selected specification.

### Select specification
- Select any of feature/ specification related to which review is to be filtered. ( any from wordcloud or simply type your querry in small case letters)
- Our model will extract all sentences related to our specification from 1000+ reviews.
- Short Review column contains summary of review for that specification only.
- It saves a lot of time reading for whole big revies to get an idea about a particular specification.
- It is a deep learning based model so it may take some time for excecution.

for battery related filtered review
![image](https://user-images.githubusercontent.com/77160352/205492062-fec9267b-f5de-48e5-ba9c-bf25a5c17928.png)
![image](https://user-images.githubusercontent.com/77160352/205492123-63f44022-bfeb-4f9b-b43b-165975b2517c.png)



### To get an idea of rating for particular specification
- We apply sentiment analysis on each of short extracted review.
- Applied sentiment analysis using TextBlob
![image](https://user-images.githubusercontent.com/77160352/205492217-04441cd5-473b-4f81-9e37-041a4296cdc2.png)


### Ask any Question related to any specification
- Ask any type of querry and our model will try to extract sentences related to mentioned querry.
- BertForQuestionAnswering is used to answer the queries
![image](https://user-images.githubusercontent.com/77160352/205492481-ec045d80-deb5-460e-8d90-019406379ec2.png)

