import cv2
import utils
import numpy as np
import progressbar
import random, os, sys, shutil
from PIL import ImageFont, ImageDraw, Image

params = utils.readConf()

fixedWidth, fixedHeight = int(params['width']), int(params['height'])
labelFile = 'images/labels.txt'

# write these parameters into a config file for thge

fonts = ['fonts/' + font for font in os.listdir('fonts') if font[-3:] == 'ttf']

if os.path.exists('images'):
    shutil.rmtree('images')
os.mkdir('images')

corpus = open('processed-corpus.txt').readlines()
labels = open(labelFile,'w')

numberOfImages = int(sys.argv[1])

def getRandomRotationAngle():
    return int(np.random.normal(0,10))

def getRandomBackgroundShade():
    '''
    random shade of gray as the bg, lighter shades have a higher probablility.
    '''
    return 255 - int(abs(np.random.normal(0,10)))

def getFizedSizeBlankPILImage(shade=255):
    '''
    Return a blank PIL.Image object with the fixed size (to be used as input
    for our network).
    The fixed size is set at the top of this file
    '''
    grayscaleKey = 'L'
    img = Image.new(grayscaleKey, size=(fixedWidth, fixedHeight),\
                                  color=shade)
    return img

def getPilImageToFitLine(text, font, shade=255):
    '''
    Returns a PIL.Image object to exactly fit the 'text', using 'font'
    '''
    width, height = font.getsize(text)
    grayscaleKey = 'L'
    img = Image.new(grayscaleKey, size=(width, height), color=shade)
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
    '''
    Generate and save a randomized training image by sampling from different
    fonts, colors and sizes of fonts, the final image will be scaled to a fixed
    size set at the top of this document. the file iamges/labels.txt will
    contain all the mappings between images and the text they contain
    '''
    if product == None:
        product = getLabelFromProductDescription()
    fill = getRandomColor()
    bgShade = getRandomBackgroundShade()
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
            img = getPilImageToFitLine(text=product, font=font)
            _, height = img.size
            if height >= 10:
                textReadable = True
            else:
                averageFontSize = averageFontSize + 1
                trys += 1

        if trys == max_trys:
            generateRandomLineImage(i)
            return

    # create and add text to an image
    img = getPilImageToFitLine(text = product, font = font, shade = bgShade)
    context = ImageDraw.Draw(img)
    context.text(xy = position, text = product, font = font, fill = fill)

    # # apply random rotation
    # randomAngle = getRandomRotationAngle()
    # img = img.rotate(randomAngle, expand=True)

    # paste the text image onto a fixed size image
    imgFixedSize = getFizedSizeBlankPILImage(shade=bgShade)
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

# generate all the images!

 # A nice progress bar to show progress
widgets = [progressbar.Percentage(),\
            progressbar.Bar(),\
            'Generating Images: ', progressbar.AnimatedMarker(markers='.oO@* ')]

batchSize = min(numberOfImages, 10000)
batches = numberOfImages//batchSize

for batch in range(batches):
    print('Batch {} of {}'.format(batch, batches))
    bar = progressbar.ProgressBar(widgets=widgets, max_value=batchSize).start()
    for i in bar(range(1,batchSize+1)):
        generateRandomLineImage(i + (batch*batchSize))
