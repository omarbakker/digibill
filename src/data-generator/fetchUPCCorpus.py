from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer
import requests
import time, random, threading, multiprocessing, sys

lock = threading.Lock()
# read txt file
corpus = open('cc.txt').readlines()

# remove all unwanted characters (\r\n)
corpus = [line.strip('\r\n') for line in corpus]

# remove duplicate entries by upc code
mapping = dict(zip([line.split(' ')[0] for line in corpus],\
                   [' '.join(line.split(' ')[1:]) for line in corpus]))


def upcdb1():
    while True:
        try:
            url = "https://www.upcdatabase.com/random_item.asp"
            result = requests.get(url)
            soup = bs(result.content, 'html.parser', parse_only=SoupStrainer('td'))
            tdtags = str(soup).replace('</td>','').split('<td>')
            description = None

            if 'Description' in tdtags:
                description = tdtags[tdtags.index('Description')+2]
                description = description.replace('\n', '').replace('\r', '')
                upcCode = tdtags[2].split("\"")[1]
            else:
                continue

            with lock:
                if upcCode not in mapping:
                    with open('corpus.txt', 'a') as corpus:
                        mapping[upcCode] = description
                        print('thread 1: ' + upcCode + ' ' + description)
                        corpus.write(upcCode + ' ' + description + '\n')
                        time.sleep(0.0  + (0.35 * random.random()))
        except:
            pass

def upcdb2():
    while True:
        try:
            url = "http://upcdatabase.org/random"
            result = requests.get(url)
            upcCode = str(result.content).split('<title>Information on barcode ')
            upcCode = upcCode[1].split(' - UPC Database</title>')[0]
            description = str(result.content).split('<h3 class="lead">')
            description = description[1].split('</h3>')[0]
            description = description.replace('\n', '').replace('\r', '')

            with lock:
                if upcCode not in mapping:
                    with open('corpus.txt', 'a') as corpus:
                        mapping[upcCode] = description
                        print('thread 2: ' + upcCode + ' ' + description)
                        corpus.write(upcCode + ' ' + description + '\n')
                        time.sleep(0.25 + (0.25 * random.random()))
        except:
            pass

db1Thread = threading.Thread(target=upcdb1).start()
# db2Thread = threading.Thread(target=upcdb2).start() # keeps getting duplicates after a day of running
