# read txt file
corpus = open('corpus.txt').readlines()

# remove all unwanted characters (\r\n)
corpus = [line.strip('\r\n') for line in corpus]

print("Total fetched items: ", len(corpus))

# remove duplicate entries by upc code
mapping = dict(zip([line.split(' ')[0] for line in corpus],\
                   [' '.join(line.split(' ')[1:]) for line in corpus]))
print("Removed ", len(corpus) - len(mapping), "duplicate codes")
corpus = [str(key + ' ' + mapping.get(key)) for key in mapping]


# remove duplicate entries by description
mapping = dict(zip([' '.join(line.split(' ')[1:]) for line in corpus],\
                   [line.split(' ')[0] for line in corpus]))
print("Removed ", len(corpus) - len(mapping), "duplicate desciptions")
print('Entries remaining: ', len(mapping))

# remove the upc codes, leaving just the words
corpus = [str(key) for key in mapping]

# remove all unwanted characters
replacements = {'&amp;' : '&',\
                r'\r\n' : '',\
                r'\n'   : '',
                '\n'   : ''}
for key in replacements:
    corpus = [line.replace(key, replacements.get(key)) for line in corpus]

# remove those stubborn linefeeds that just wont go away
for i in range(len(corpus)):
    ''.join(corpus[i].split(chr(10)))


# write to output file
with open('processed-corpus.txt', 'w') as out:
    out.write('\n'.join(corpus))

# bag of words analysis
words = ' '.join(corpus).split(' ')
wordSet = set(words)
countMap = dict(zip(list(wordSet), [0] * len(wordSet)))
for word in words:
    countMap[word] = countMap[word] + 1
countMap = list(countMap.items())
countMap.sort(key = lambda tup:tup[1], reverse = True)

# print the most occurring 100 words
for i in range(100):
    print(countMap[i])
