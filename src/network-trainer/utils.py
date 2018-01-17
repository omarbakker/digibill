import numpy as np
import cv2

conf = open('../data-generator/params.conf').readlines()
params = [line.split(' = ') for line in conf if line[0] is not '#']
params = [param for param in params if len(param) is 2]
params = dict(zip([p[0].strip() for p in params],\
                  [p[1].strip() for p in params]))

imgDir = '../data-generator/images/'
imgWidth = int(params['width'])
imgHeight = int(params['height'])
channels = 1

testPercentage = 0.05
batchSize = 50
labels = open(imgDir + 'labels.txt').readlines()
labels = [l.rstrip('\n').rstrip(' ') for l in labels]
labels = [l.split(':') for l in labels]
images = [l[0].rstrip() for l in labels]
testImgCount = int(len(images) * testPercentage)
testImages = images[-testImgCount:]
images = images[:-testImgCount]
print('Image count: {}'.format(len(images) + len(testImages)))
labelMap = dict(zip(images, [l[1].rstrip() for l in labels]))

def shuffle(lst):
    return list(np.array(lst)[np.random.permutation(len(lst))])

def encodeLabel(label):
    return [encodings.get(c) for c in list(label) if c in encodings]

def getLabel(img):
    if img not in labelMap: print("Error: {} label not found".format(img))
    return labelMap.get(img)

def readImage(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    img = np.reshape(img, [imgHeight, imgWidth, channels])
    return img

def batches():
    shuffle(images);
    batchSize = batchSize
    numbatches = images // batchSize
    for i in range(numbatches):
        batchImgs = images[i * batchSize: (i+1) * batchSize]
        encodedLabels = [encodeLabel(getLabel(img)) for img in batchImgs]
        batchImgs = [readImage(img) for img in batchImgs]
        yield batchImgs, encodedLabels



def getOutputMaps(verbose=False):
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
    encodings = dict(zip([c[0] for c in countsList if c[0] is not ' '],
                        range(1, len(countsList))))
    decodings = dict(zip(range(1, len(countsList)),
                        [c[0] for c in countsList if c[0] is not ' ']))
    encodings[' '] = 0
    decodings[0] = ' '
    return encodings, decodings

def dictToString(d):
    items = [':'.join([str(i) for i in item]) for item in list(d.items())]
    return '\n'.join(items)

def saveOutputMaps(encodings, decodings):
    with open('../data-generator/encodings.txt', 'w') as enc:
        enc.write(dictToString(encodings))
    with open('../data-generator/decodings.txt', 'w') as dec:
        dec.write(dictToString(decodings))

encodings, decodings = getOutputMaps()
nClasses = len(encodings) - 1 # -1 because space is not a class
saveOutputMaps(encodings, decodings)
