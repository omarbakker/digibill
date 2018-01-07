import cv2
import numpy as np
import random, os, sys
from PIL import ImageFont, ImageDraw, Image

fonts = ['fonts/' + font for font in os.listdir('fonts') if font[-3:] == 'ttf']

if not os.path.exists('images'):
    os.mkdir('images')

corpus = open('processed-corpus.txt').readlines()
labels = open('images/labels.txt','w')

numberOfImages = int(sys.argv[1])

def getRandomBlankNumpyImage():
    '''
    return a white image with random width, and a constant height of 50
    '''
    channels = 1
    height = 28
    width = int(50 + random.random() * 200)
    img = np.ones((height, width, channels), dtype=np.uint8) * 255
    return img

def getFizedSizeBlankPILImage():
    height = 28
    width = 320
    grayscaleKey = 'L'
    white = 255
    img = Image.new(grayscaleKey, size=(width, height), color=white)
    return img

def getPilImageToFitLine(text, font):
    width, height = font.getsize(text)
    grayscaleKey = 'L'
    white = 255
    img = Image.new(grayscaleKey, size=(width, height), color=white)
    return img

def getRandomColor():
    '''
    Return a random color from grayscale,
     (darker shades have a higher probablility)
    '''
    return int(abs(np.random.normal(0,0.25)) * 255)

def getRandomFont(averageSize=14):
    '''
    Return a random font from the fonts directory
    Add more fonts to sample from by placing them in data-generator/fonts
    '''
    randomIndex = random.randint(0,len(fonts)-1)
    randomSize = max(6,abs(int(np.random.normal(averageSize,4))))
    # print('size: {}'.format(randomSize))
    return ImageFont.truetype(fonts[randomIndex], randomSize)

def getRandomProductDescription():
    '''
    Returns a random product from the udp database
    to be used by getLabelFromProductDescription()
    '''
    randomIndex = random.randint(0,len(corpus)-1)
    randomProduct = corpus[randomIndex]
    randomProduct = randomProduct.split(' ')
    return randomProduct

def getLabelFromProductDescription():
    '''
        Return a random string from the list of product descriptions
        most of the time we return one word
        but since our segmentation algorithm might mistake two words for one,
        we should include some grouped words in the same training image
    '''
    product = getRandomProductDescription()
    numberOfWords =  1 + int(abs(np.random.normal(0,0.75)))
    while len(product) < numberOfWords:
        numberOfWords = numberOfWords - 1
    if len(product) == numberOfWords:
        randomIndex = 0
    else:
        randomIndex = random.randint(0, len(product)-numberOfWords)
    randomWord = ' '.join(product[randomIndex:randomIndex+numberOfWords])
    return randomWord.rstrip()

def generateRandomLineImage(i, product=None, font=None):
    if product == None:
        product = getLabelFromProductDescription()
    fill = getRandomColor()
    position = (0,0)
    textReadable = False
    averageFontSize = 12
    img = None
    max_trys, trys = 10, 1

    # since fonts usually have different real sizes for the same specified
    # text size, make sure the text is readable by ensuring image height > 10
    if font == None:
        while not textReadable and trys < max_trys:
            font = getRandomFont(averageSize=averageFontSize)
            img = getPilImageToFitLine(text = product, font = font)
            _, height = img.size
            if height >= 10:
                textReadable = True
            else:
                averageFontSize = averageFontSize + 1
                trys += 1

        if trys == max_trys:
            generateRandomLineImage(i)
            return

    img = getPilImageToFitLine(text = product, font = font)
    context = ImageDraw.Draw(img)
    context.text(xy = position, text = product, font = font, fill = fill)
    imgFixedSize = getFizedSizeBlankPILImage()
    fixedWidth, fixedHeight = imgFixedSize.size
    oldWidth, oldHeight = img.size
    newWidth = int(oldWidth * fixedHeight/oldHeight)
    if newWidth <= fixedWidth:
        textFitsImage = True
        img = img.resize((newWidth, fixedHeight))
        imgFixedSize.paste(img, box = (0,0,newWidth,fixedHeight))
    else:
        # the word is too long and needs to be broken up
        split = len(product) // 2
        generateRandomLineImage('{}-2'.format(i),\
                                product=product[split:],\
                                font=font)
        generateRandomLineImage(i, product=product[:split], font=font)
        return


    imgFixedSize.save('images/image{}.png'.format(i))
    labels.write('image{}.png: {}\n'.format(i, product))


for i in range(1,numberOfImages+1):
    generateRandomLineImage(i)
    print(i)
