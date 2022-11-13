from bs4 import BeautifulSoup

soup = BeautifulSoup(sentence)
sentence = soup.get_text()