import utils
import tensorflow as tf
from utils import prints

learningRate = 1e-4
epochs = 3 # since we can generate as much data as we need
batchSize = utils.batchSize
lstmSize = 2**9
pkeepi = 0.5
pkeepConvi = 0.7
pkeepLSTMi = 0.8
seqlen = utils.seqlen
logdir = './log'
checkpointdir = './checkpoint'
restore = True # restore from checkpoint
nClasses = utils.nClasses

params = utils.params
imgWidth = int(params['width'])
imgHeight = int(params['height'])
channels = 1

# this will hold the input to the CNN
x = tf.placeholder(tf.float32, [None, imgWidth * imgHeight])
x = tf.reshape(x, [-1, imgHeight, imgWidth, channels])

# this will hold the output of the entire network
y = tf.sparse_placeholder(tf.int32)
yDense = tf.placeholder(tf.int32, [None, 50])
yDense = tf.reshape(yDense, [-1, 50])

# learning rate placeholder
lr = tf.placeholder(tf.float32)

# test flag for batch normalization
tst = tf.placeholder(tf.bool)
itera = tf.placeholder(tf.int32)

# dropout probs
pkeep = tf.placeholder(tf.float32)
pkeepConv = tf.placeholder(tf.float32)
pkeepLSTM = tf.placeholder(tf.float32)

# list to hold moving exponential moving average ops
ema = []

def relu(x_in):
    return tf.nn.relu(x_in)

def dropout(x_in, isConv=False):
    if isConv:
        return tf.nn.dropout(x_in, pkeepConv, compatibleConvNoiseShape(x_in))
    else:
        return tf.nn.dropout(x_in, pkeep)

def conv(x_in, channels, filters, filterShape, name):
    convFilterShape = [filterShape[0], filterShape[1], channels, filters]
    w = tf.Variable(tf.truncated_normal(convFilterShape, stddev=0.03),
                                      name=name+'_W')
    b = tf.Variable(tf.truncated_normal([filters]), name=name+'_b')
    out = tf.nn.conv2d(x_in, w, strides=[1,1,1,1], padding='SAME')
    return out + b, b

def pool(x_in, shape, strides):
    ksize = [1, shape[0], shape[1], 1]
    return tf.nn.max_pool(x_in, ksize=ksize, strides=strides, padding='SAME')

def dense(x_in, shape, name):
    w_i = tf.truncated_normal(shape, stddev=0.03)
    b_i = tf.truncated_normal([shape[-1]], stddev=0.01)
    w = tf.Variable(w_i, name=name+'_W')
    b = tf.Variable(b_i, name=name+'_b')
    return tf.matmul(x_in, w) + b, b

def batchNorm(x_in, isTest, iteration, offset, isConv=False):
    # adapted from sarvesh278 kernel on kaggle
    emv = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if isConv:
        mean, variance = tf.nn.moments(x_in, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(x_in, [0])
    update_moving_everages = emv.apply([mean, variance])
    m = tf.cond(isTest, lambda: emv.average(mean), lambda: mean)
    v = tf.cond(isTest, lambda: emv.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(x_in, m, v, offset, None, bnepsilon)
    ema.append(update_moving_everages)
    return Ybn

def compatibleConvNoiseShape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1])\
                            + tf.constant([0,1,1,0])
    return noiseshape

def bilstm(x_in, sequence_length, rnn_size, scope):
    """Build bidirectional (concatenated output) RNN layer"""

    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    lstmCell = tf.contrib.rnn.LSTMCell
    DropoutWrapper = tf.contrib.rnn.DropoutWrapper
    cell_fw = lstmCell(rnn_size, initializer=weight_initializer)
    cell_bw = lstmCell(rnn_size, initializer=weight_initializer)
    cell_fw = DropoutWrapper(cell_fw, input_keep_prob=pkeepLSTMi)
    cell_bw = DropoutWrapper(cell_bw, input_keep_prob=pkeepLSTMi)

    rnn_output, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, x_in,
        sequence_length=[seqlen]*batchSize,
        dtype=tf.float32,
        scope=scope)

    outfw, outbw = rnn_output
    return tf.concat([outfw, outbw], axis=2,name='output_stack')
    # return rnn_output

def convolutionalLayers(x_in):
    #1 conv -> batchnorm -> relu -> dropout -> pool
    conv1, b1 = conv(x_in, 1, 64, [5,5], name='conv1')
    norm1 = batchNorm(conv1, tst, itera, b1, isConv=True)
    relu1 = relu(norm1)
    drop1 = dropout(relu1, isConv=True)
    pool1 = pool(drop1, shape=[3,3], strides=[1, 3, 3, 1])

    #2 conv -> batchnorm -> relu -> dropout -> pool
    conv2, b2 = conv(pool1, 64, 128, [5,5], name='conv2')
    norm2 = batchNorm(conv2, tst, itera, b2, isConv=True)
    relu2 = relu(norm2)
    drop2 = dropout(relu2, isConv=True)
    pool2 = pool(drop2, shape=[3,3], strides=[1, 3, 3, 1])

    #3 conv -> batchnorm -> relu -> dropout -> pool
    conv3, b3 = conv(pool2, 128, 96, [5,5], name = 'conv3')
    norm3 = batchNorm(conv3, tst, itera, b3, isConv=True)
    relu3 = relu(norm3)
    drop3 = dropout(relu3, isConv=True)
    pool3 = pool(drop3, shape=[2,2], strides=[1, 2, 2, 1])
    features = tf.reshape(pool3, [-1, 16 * 2 * 96])

    return features

def denseLayers(x_in):
    #4 dense -> batchnorm -> relu -> dropout
    dense1, bd1 = dense(x_in, shape=[16 * 2 * 96, 1000], name='dense1')
    normd1 = batchNorm(dense1, tst, itera, bd1)
    relud1 = relu(normd1)
    dropd1 = dropout(relud1)

    #5 dense -> batchnorm -> relu -> dropout
    dense2, bd2 = dense(dropd1, shape=[1000, seqlen], name='dense2')
    normd2 = batchNorm(dense2, tst, itera, bd2)
    relud2 = relu(normd2)
    dropd2 = dropout(relud2) # seq len = 100 ?

    return dropd2

def recurrentLayers(x_in):

    x_in = tf.reshape(x_in, [batchSize, seqlen, 1])

    #6 bilstm -> bilstm -> dense -> relu
    lstm1 = bilstm(x_in, seqlen, lstmSize, "bilstm1")
    lstm2 = bilstm(lstm1, seqlen, lstmSize, "bilstm2")

    # reshaping to apply weights over timesteps
    lstm2 = tf.reshape(lstm2, [-1, lstmSize * 2])
    out, _ = dense(lstm2, [lstmSize * 2, nClasses + 1], 'out')

    shape = tf.shape(x_in)
    y_ = tf.reshape(relu(out), [shape[0], -1, nClasses + 1])
    y_ = tf.transpose(y_, (1, 0, 2))
    return y_

def CTCLoss(y_in, y_):
    #7 Connectionist Temporal Classification
    loss = tf.nn.ctc_loss(labels=y_in, inputs=y_,
                            sequence_length=[seqlen]*batchSize, time_major=True)
    meanLoss = tf.reduce_mean(loss)
    return meanLoss

def trainSteps(loss):
    step = tf.Variable(0, trainable=False)
    return tf.train.AdamOptimizer(lr).minimize(loss, global_step=step)

def evaluationSteps(logits, loss):
    decoder = tf.nn.ctc_beam_search_decoder
    decoded, _ = decoder(logits, [seqlen] * batchSize,
                         merge_repeated=False, top_paths=1)
    denseDecoded = tf.sparse_tensor_to_dense(decoded[0], default_value=0)
    denseDecoded = tf.cast(denseDecoded, tf.int32)
    outputLength = tf.minimum(tf.constant(50), tf.shape(denseDecoded)[1])
    denseReshaped = tf.Variable(tf.zeros([batchSize, 50], tf.int32))
    denseReshaped = tf.assign(denseReshaped[:,:outputLength], denseDecoded[:,:outputLength])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(denseReshaped, yDense), tf.float32))
    lossSummary = tf.summary.scalar('batch_loss', loss)
    accuracySummary = tf.summary.scalar('batch_accuracy', accuracy)
    summaries = tf.summary.merge([lossSummary, accuracySummary])
    return accuracy, denseDecoded, summaries

def buildModel(x_in, y_in):
    features = convolutionalLayers(x_in)
    sequenceIn = denseLayers(features)
    logits = recurrentLayers(sequenceIn)
    loss = CTCLoss(y_in, logits)
    trainStep = trainSteps(loss)
    accuracy, denseDecoded, summaries = evaluationSteps(logits, loss)
    return trainStep, denseDecoded, accuracy, loss, summaries

if __name__ == '__main__':
    with tf.Session() as sess:

        trainOp, y_out, accuracyOp, lossOp, summariesOp = buildModel(x, y)
        ema = tf.group(ema)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        writer = tf.summary.FileWriter(logdir + '/train', sess.graph)

        if restore:
            chkpt = tf.train.latest_checkpoint(checkpointdir)
            if chkpt:
                print('restoring checkpoint: {}'.format(chkpt))
                saver.restore(sess, chkpt)

        prints('Training Begins!')

        for epochNum in range(epochs):
            prints('Starting epoch {}'.format(epochNum+1))
            batchNum = 0

            for batch, labels, denseLabels in utils.batches():
                batchNum += 1
                iteration = (batchSize*(epochNum)) + batchNum

                feed = {x: batch, y:labels, yDense:denseLabels,
                        lr:learningRate, pkeep:pkeepi, pkeepConv:pkeepConvi,
                        pkeepLSTM:pkeepLSTMi, tst:False}
                feedema = {x: batch, y: labels, tst: False, itera: iteration,
                            pkeep: 1.0, pkeepConv: 1.0, pkeepLSTM:1.0}

                _, preds, accuracy, loss, summaries = sess.run([trainOp, y_out,
                                                          accuracyOp, lossOp,
                                                           summariesOp], feed)
                sess.run(ema, feedema)
                writer.add_summary(summaries)

                label = utils.decodedLabel(denseLabels[0]).strip()
                pred = utils.decodedLabel(preds[0]).strip()
                print('Batch {:05}, loss {:3.2f}, accuracy: {:3.2f}, sample: {:^20}  |  {:^20}'.
                        format(batchNum, loss, accuracy*100, label, pred))






# end
