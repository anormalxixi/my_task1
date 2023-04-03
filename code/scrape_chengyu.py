import requests
from bs4 import BeautifulSoup
response = requests.get("https://hanyu.baidu.com/s?wd=五光十色&cf=jielong&ptype=idiom")
html = response.text
soup = BeautifulSoup(html, "html.parser")
jielong = soup.findAll("div", id="jielong-wrapper")
for a in jielong:
    idioms = a.findAll("a")
for idiom in idioms[:20]:
    idiom = idiom.text
    response = requests.get(f"https://hanyu.baidu.com/s?wd={idiom}&cf=jielong&ptype=idiom")
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    word = soup.findAll("strong")
    print(word[0].string)
    explain = soup.find("p")
    print(explain.string)
    syn_ant = soup.findAll("div", attrs="syn_ant")
    for i in syn_ant:
        labels = i.findAll("label")
        for label in labels:
            print(label.text)
            synandant = soup.findAll("div", attrs="block")
            if label.text == "近义词 ":
                print(synandant[0].text)
            if label.text == "反义词":
                print(synandant[1].text)