import requests
from bs4 import BeautifulSoup
response = requests.get(f"https://hanyu.baidu.com/s?wd=无赖&cf=hot&ptype=zici")
html = response.text
soup = BeautifulSoup(html, "html.parser")
list = soup.findAll("li", attrs="recmd-item no-img")
names = []
for name in list:
    names.append(name.text)
for each in ["\n智能\n", "\n氤氲\n", "\n徘徊\n", "\n感恩\n", "\n谦虚\n", "\n憧憬\n", "\n期待\n", "\n刹那\n"]:
    names.append(each)

for name in names:
    response = requests.get(f"https://hanyu.baidu.com/s?wd={name}&cf=hot&ptype=zici")
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    word = soup.findAll("strong")
    print(word[0].text)
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
            else:
                print(synandant[1].text)











