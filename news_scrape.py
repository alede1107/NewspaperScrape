from newspaper import Article
import requests
import nltk
nltk.download('punkt')

def summarize_article(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36'
    headers = {'User-Agent': user_agent}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch article.")
        return None
    html = response.text
    article = Article(url)
    article.set_html(html)
    article.parse()
    if not article.text.strip():
        print("No text found. Cannot summarize.")
        return None
    article.nlp()
    summary=article.summary


    print("\n--- ARTICLE INFO ---")
    print("Author(s): " + str(article.authors))
    date = article.publish_date
    if date:
        print("Publish Date: " + str(date.strftime("%m/%d/%y")))
    else:
        print("Publish Date: N/A")

    print("Top Image URL: " + str(article.top_image))
    image_string = "All Images: "
    for image in article.images:
        image_string+="\n\t" + image
    print(image_string)

    print("A Quick Article Summary")
    print("-----------------------------------")
    print(summary)
    return summary

if __name__ == "__main__":
    url = input("Enter the full article URL: ").strip()
    if "?" in url:
        url = url.split("?")[0]
    summary = summarize_article(url)