
# coding: utf-8

# 2017-06-19
# Read sentence
# Word2Vec
# With the sentence, calculate it's "Probability"
# 

# 

# In[7]:

import gensim


# In[1]:

joke_text = []


# In[5]:

#from http://onelinefun.com/puns/
joke_text.append("I wasn't originally going to get a brain transplant, but then I changed my mind.")
joke_text.append("I was going to get a brain transplant and then I changed my mind.")
joke_text.append("I'd tell you a chemistry joke but I know I wouldn't get a reaction.")
joke_text.append("I'm glad I know sign language, it's pretty handy.")
joke_text.append("I have a few jokes about unemployed people but it doesn't matter none of them work.")
joke_text.append("I used to be a banker, but then I lost interest.")
joke_text.append("I hate insects puns, they really bug me.")
joke_text.append("Insect puns bug me.")
joke_text.append("It's hard to explain puns to kleptomaniacs because they always take things literally.")
joke_text.append("I was so sad and crying when I lost my playstation 3 but unfortunately, there was nobody to console me!")
joke_text.append("I'm on a whiskey diet. I've lost three days already.")

#from http://www.jokesclean.com/Puns/
joke_text.append("She broke into song when she couldn't find the key.")
joke_text.append("She had a boyfriend with a wooden leg, but broke it off.")
joke_text.append("Corduroy pillows are making headlines.")

#from http://thoughtcatalog.com/christopher-hudspeth/2013/09/50-terrible-quick-jokes-thatll-get-you-a-laugh-on-demand/2/
joke_text.append("If you want to catch a squirrel just climb a tree and act like a nut.")
joke_text.append("A magician was walking down the street and turned into a grocery store.")
joke_text.append("Time flies like an arrow, fruit flies like banana.")
joke_text.append("Dwarfs and midgets have very little in common.") # appropriate?

#from Reddit (https://www.reddit.com/r/AskReddit/comments/1sy48v/whats_your_favourite_oneliner_pun/)
joke_text.append("My grandfather has the heart of a lion, and a lifetime ban at the zoo.")
joke_text.append("A broken pencil is pointless.")
joke_text.append("My skiing skills are really going downhill.")
joke_text.append("Obesity should not be taken lightly.")


# In[6]:

joke_text


# In[ ]:



