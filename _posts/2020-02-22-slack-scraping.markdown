---
layout: post
title:  Data Scraping off a Private Slack Channel
date:   2020-02-22 23:21:21 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: post-2.jpg # Add image post (optional)
tags: [Blog, Web Scraping, Data Scraping, Data Analysis]
author: James Xuoi # Add name author (optional)
urlcolor: green
--- 

It has been a while since I published my 1st post, 'speech recognition, part 1'. Before finishing writing part 2, I'd like to freshen up my blog with a mini Data Science project, 'Slack Data Scraping' from scartch for beginners with no experience required. 

On the final day of my Data Science course, our class were invited out and got treated delicious milk shakes by our Lead Trainer, Dr Chaintanya Rao, an ex-Data Scientist & Researcher at IBM and Testra, he is going to be the lecturer at Melbourne University, Top 1 University in Australia next couple of months (A big congrats to you again if you are reading this post!). While waiting for our milk shakes to get done, Chaitanya asked for a volunteer to scrape all data off our Priviate Slack channel, where we often shared and stored our learning materials. Even though I have only done data scraping for once or twice, while waiting for a hand to raise in an odd almost sphere with eyes avoided looking at each others, I decided to raise my hand and take the job to enhance my data scraping skill, and thats how I started the project.


> **What is Data Scraping?**

![Data Scraping]({{site.baseurl}}/assets/img/post_2/Data-Scraping.jpg)

Data scraping refers to a technique in which a computer program extracts data from output generated from another program. Data scraping is commonly manifest in web scraping, the process of using an application to extract valuable information from a website.

> **Slack Data Scrapping**

Slack is a cloud-based instant messaging application, which is commonly used as a communication platform between co-workers. There are 3 types of channel: Public channels, Private channels, Direct messages. This post introduces how to scrape data off a Slack Private channel.  

Before going through the steps below, it is a MUST to have a Slack account, which has joined at least 1 Slack Channel in order to start scraping data. If not, you need to complete it before continuing reading.

### 1. Signing in your current Slack account > Go to ['https://api.slack.com/apps'](https://api.slack.com/apps) > Click on 'Create New App'.

![Data Scraping]({{site.baseurl}}/assets/img/post_2/1.jpg)

### 2. Filling up your 'App_Name' and choose the current 'Slack Work Space', where contains the channel that you'd like to scrap data from.

![Data Scraping]({{site.baseurl}}/assets/img/post_2/2.jpg)

### 3. Go to 'OAuth & Permissions' > Under 'Scopes', click on 'Add an OAuth Scope' to 'User Token Scopes'. 

![Data Scraping]({{site.baseurl}}/assets/img/post_2/4.jpg)

###    Scroll up and click 'Install App to Work space'.

![Data Scraping]({{site.baseurl}}/assets/img/post_2/5.jpg)

###    Your 'OAuth Access Token' and 'Bot User OAuth Access Token' will be automatically generated. Copy and paste it to a txt file; then save it to somewhere safe!

![Data Scraping]({{site.baseurl}}/assets/img/post_2/6.jpg)

   * To find out more about the permission of each OAuth Scope, you can copy and paste the scope's name on the search bar of [Slack Documentation page](https://api.slack.com/#read_the_docs) for more information.

### 4. Install python 'Slack Client' package with 'pip install slackclient', find out more about documentation of the library [here](https://pypi.org/project/slackclient/)

![Data Scraping]({{site.baseurl}}/assets/img/post_2/7.jpg)

### 5. Open your favourite Python working platform: Jupyter Notebook/Spyder/Google Colab/etc. In this post, I use Jupyter Notebook as it is the most commonly used platform.

### 6. Import the nesscesary libraries and connect 'Slack Client' with your 'OAuth Access Token' or 'Bot User OAuth Access Token'. 


   * 'Bot User OAuth Access Token' is commonly used for Chat Bot development. Using that token shows your App_Name on the Slack Work Space and it need to seek for permission from the channel's admins to gain access.
   * 'OAuth Access Token' allows you to scrape data without seeking for any permission if you're already a member of the group.

### 7. Scrap the conversation's history on a specific channel by inserting the channel code, which can be found at the end of the link.

![Data Scraping]({{site.baseurl}}/assets/img/post_2/8.jpg)

``` python
# Connect 'Slack Client' with your 'OAuth Access Token' 
os.environ['SLACK_API_TOKEN'] = 'OAuth_Access_Token'
slack_token = os.environ["SLACK_API_TOKEN"]
sc = SlackClient(slack_token)
```
``` python
a = sc.api_call( "conversations.history",channel="Channel_ID")
a.keys()
b = a.get('messages')
f = []
for i in b:
    for e in i:
        if e == 'attachments':
            f.append(i)
```

###    Covert data to pandas dataframe then save it as .csv file or continue to perform data exploratory analysis.

``` python
df = pd.DataFrame(f)
```
![Data Scraping]({{site.baseurl}}/assets/img/post_2/9.jpg)


### Mini Demo of Slack Text Visualisation.

![Data Scraping]({{site.baseurl}}/assets/img/post_2/10.jpg)

   * View details of the source code on my Github page [here.]()




