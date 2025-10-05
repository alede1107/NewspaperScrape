from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
def calculate_average(list):
    return sum(list)/len(list)
def calculate_sentiment(sentiment, type):
    sentiment_category = ""
    if type == "polarity":
        if sentiment > 0.75:
            sentiment_category = "Extremely Positive"
        elif sentiment >0.5:
            sentiment_category = ("Significantly Positive")
        elif sentiment > 0.3:
            sentiment_category = ("Fairly Positive")
        elif sentiment > 0.1:
            sentiment_category = ("Slightly Positive")
        elif sentiment <-0.1:
            sentiment_category = ("Slightly Negative")
        elif sentiment < -0.3:
            sentiment_category = ("Fairly Negative")
        elif sentiment < -0.5:
            sentiment_category = ("Significantly Negative")
        elif sentiment < -0.75:
            sentiment_category = ("Extremely Negative")
        else:
            sentiment_category = ("Neutral")
        return sentiment_category
    elif type == "subjectivity":
        if sentiment > 0.75:
            sentiment_category = "Extremely Subjective"
        elif sentiment >0.5:
            sentiment_category = ("Fairly Subjective")
        elif sentiment > 0.3:
            sentiment_category = ("Fairly Objective")
        elif sentiment > 0.1:
            sentiment_category = ("Extremely Objective")
        else:
            sentiment_category = "Objective"
        return sentiment_category
    else:
        print("Invalid Input")

def find_sentiment(news_story):
    news = TextBlob(news_story)
    sentiments = []
    for sentence in news.sentences:
        sentiment = sentence.sentiment
        for metric in sentiment:
            sentiments.append(metric)

    polarity_data = []
    subjectivity_data = []
    for i in range(len(sentiments)):
        if i%2==0:
            polarity_data.append(sentiments[i])
        else:
            subjectivity_data.append(sentiments[i])
    polarity_average = calculate_average(polarity_data)
    subjectivity_average = calculate_average(subjectivity_data)
    print()
    print("TEXTBLOB ANALYSIS")
    print("--------------------------------")
    print("Polarity: "+ calculate_sentiment(polarity_average, "polarity"))
    print("Subjectivity: "+ calculate_sentiment(subjectivity_average, "subjectivity"))

def vader_sentiment(news_story):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(news_story)

    compound = scores['compound']
    if compound >= 0.05:
        sentiment_label = "Positive"
    elif compound <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    print("\nVADER ANALYSIS")
    print("--------------------------------")
    print("Compound Score: " + str(round(compound, 3)))
    print("Overall Sentiment: " + sentiment_label)
    print("Breakdown: "
          + "Positive=" + str(round(scores['pos'], 3))
          + " | Neutral=" + str(round(scores['neu'], 3))
          + " | Negative=" + str(round(scores['neg'], 3)))


if __name__ == "__main__":
    print("Enter the article text below (paste summary or paragraph):\n")
    user_input = ""
    while True:
        line = input()
        if line == "":
            break
        user_input += line + " "

    if user_input.strip():
        find_sentiment(user_input)
        vader_sentiment(user_input)
    else:
        print("No text entered. Exiting.")