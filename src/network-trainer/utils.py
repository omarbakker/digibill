

def readConf():
    conf = open('../data-generator/params.conf').readlines()
    params = [line.split(' = ') for line in conf if line[0] is not '#']
    params = [param for param in params if len(param) is 2]
    params = dict(zip([p[0].strip() for p in params],\
                      [p[1].strip() for p in params]))
    return params

def getOutputEncodings(verbose=False):
    corpus = open('../data-generator/processed-corpus.txt').read().split('\n')
    corpus = list(' '.join(corpus))
    corpus = [char for char in corpus if char is not '\t']
    unique = set(corpus)

    mapping = dict(zip(unique, [0] * len(unique)))
    for c in corpus:
        mapping[c] = mapping[c] + 1
    countsList = list(mapping.items())
    countsList.sort(key = lambda tup:tup[1], reverse = True)
    if verbose:
        print('Top 50 chars:')
        for i in range(50):
            print(countsList[i])
        print(len(countsList))
    mapping = dict(zip([c[0] for c in countsList],\
                        range(len(countsList))))
    return mapping
