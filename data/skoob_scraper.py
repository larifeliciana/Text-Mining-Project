#import webbrowser

#webbrowser.open("https://www.amazon.com.br/revolu%C3%A7%C3%A3o-dos-bichos-conto-fadas/product-reviews/8535909559/")
import bs4
import urllib3.request as requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



def get_soup(url):
    import urllib.request

    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    x= (response.read())

    return bs4.BeautifulSoup(x)

def get(type, url):

    soup = get_soup(url)
    link = get_type(type, soup)

    if link!= None:
        lista = get_reviews(link)
    else: lista = None

    return lista




def save_list(lista,endereco):
    arq = open(endereco, 'a', encoding="utf-8")
    if lista != None:
        for i in lista:
                arq.writelines(i+"\n")



def get_rate(url):
    soup = get_soup(url)
    elemento = str(soup.find_all('star-rating'))
    open("teste.txt", 'w', encoding='utf-8').write(elemento)
    lista = []
    for i in elemento.split("rate=\""):
        lista.append(i[0])
    return lista[2:]

def get_review(url):
    soup = get_soup(url)
    elemento = str(soup.find_all('div', {'class': 'curva2-5'})).split('<div id=\"resenha')
    lista = []
    for element in elemento[1:]:
            # if element[0] != '<':
            lista.append(pre_review(element))

    return lista
def pre_review(texto):
    start = '<br/>'
    end = '</div>'
    x1 = texto.split(start,1)
    x = x1[1].split(end)
    import re
    x =  re.sub('<[^>]*>', " ", x[0])
    x = re.sub('\n',' ',x)
    x = re.sub('\r',' ',x)
    return x

def checar_page(url):
    x = get_soup(url).text
    if "Nenhum conteúdo encontrado." in x:
        return True
    return False
def get_reviews(url): #https://www.skoob.com.br/livro/resenhas/247555/edicao:277187/mpage:1
    pos = []
    neg = []
    conta = 1
    x = True
    while x:
        rate = get_rate(url)
        lista = get_review(url)
        for i in range(len(rate)):
            if int(rate[i]) == 5 or int(rate[i]) == 4 or int(rate[i]) == 3:
                pos.append(lista[i])
            elif int(rate[i]) == 0 or int(rate[i]) == 1  or int(rate[i]) == 2:
                neg.append(lista[i])
        conta = conta + 1
        print(conta)
        url = url[:-1]+ str(conta)
        if checar_page(url):
            x = False
            
    return pos, neg

def get_type(type, soup): #pos or neg

    if type is "pos":
        link = soup.find_all(class_="a-size-small a-link-normal 5star")
    else:
        link =  soup.find_all(class_="a-size-small a-link-normal 1star")
    try:
        return "https://www.amazon.com.br"+ str(link).split("\"")[5]
    except: return None

def get_next_page(soup):

    x = soup.find_all(class_="a-last")

    try:
        link = "https://www.amazon.com.br" + str(x).split("\"")[3]
    except:
        print("acabou")
        link = None
    print(link)
    if link != "https://www.amazon.com.bra-letter-space" and link != None:
       return link.replace(";", "&")

    else: return None
