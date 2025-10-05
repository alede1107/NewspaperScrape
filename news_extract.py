import requests
from bs4 import BeautifulSoup as soup
import re

def get_content_string(url):
    # get outline of all html from webpage
    page = requests.get(url)
    page_soup = soup(page.content, 'html.parser')

    # find all containers that match html we want to extract
    containers = page_soup.find_all("script", {"type": "application/ld+json"})

    # tiny snippet of html outline including all hyperlinks that we want to extract
    article_list = []
    for container in containers:
        if container.string:
            article_list.append(container.string)
    if not article_list:
        print("No JSON-LD scripts found on this page.")
        return None
    # pulled first element of article list because the rest were duplicates
    article_list[0:2] = [''.join(article_list[0:2])]
    content_string = article_list[0]

    # extract all the urls and metadata needed
    article_index = content_string.find("itemListElement")
    if article_index!= -1:
        content_string = content_string[article_index+18:]
    # took a substring of content string with article index to create final library
    # and content string of article hyperlinks
    if article_index !=-1:
        content_string = content_string[article_index + 18:]
    return content_string

def find_occurrences(content_string):
    if not content_string:
        return [], []

        # universal URL pattern
    pattern = r'https?://[^\s"\']+'
    matches = list(re.finditer(pattern, content_string))

    start_indices = [m.start() for m in matches]
    end_indices = [m.end() for m in matches]

    return start_indices, end_indices


def get_all_url(start_indices, end_indices, content_string):
    url_list = []
    for i in range(len(start_indices)):
        url_list.append(content_string[start_indices[i]:end_indices[i]])
    return url_list

if __name__ == "__main__":
    url = input("Enter the full article URL: ").strip()
    html_content = get_content_string(url)
    if html_content:
        print("Successfully fetched content!")
    else:
        print("Failed to fetch content.")
