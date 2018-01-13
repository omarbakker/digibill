import utils
import tensorflow as tf

learningRate = 1e-4
epochs = 3 # since we can generate as much data as we need
batchSize = 50
lstmSize = 2**9

params = utils.readConf()
imgWidth = int(params['width'])
imgHeight = int(params['height'])

charMap = utils.getOutputEncodings()

# this will hold the input to the CNN
x = tf.placeholder(tf.float32, [None, imgWidth * imgHeight])
xShaped = tf.reshape(x, [-1, imgHeight, imgWidth, 1])

# this will hold the output of the entire network
y = tf.placeholder(tf.float32, [None, len(charMap)])

# learning rate placeholder
lr = tf.placeholder(tf.float32)

# test flag for batch normalization
tst = tf.placeholder(tf.bool)
itera = tf.placeholder(tf.int32)

# dropout probs
pkeep = tf.placeholder(tf.float32)
pkeepConv = tf.placeholder(tf.float32)
pkeepLSTM = tf.placeholder(tf.float32)

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
    w = tf.variable(w_i, name=name+'_W')
    b = tf.variable(b_i, name=name+'_b'))
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
    return Ybn, update_moving_everages

def compatibleConvNoiseShape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

def bilstm(bottom_sequence,sequence_length,rnn_size,scope):
    """Build bidirectional (concatenated output) RNN layer"""

    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    lstmCell = tf.contrib.rnn.LSTMCell
    DropoutWrapper = tf.contrib.rnn.DropoutWrapper
    cell_fw = lstmCell(rnn_size, initializer=weight_initializer)
    cell_bw = lstmCell(rnn_size, initializer=weight_initializer)
    cell_fw = DropoutWrapper(cell_fw, input_keep_prob=dropout_rate )
    cell_bw = DropoutWrapper(cell_bw, input_keep_prob=dropout_rate )

    rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)
    return tf.concat(rnn_output,2,name='output_stack')

#1 conv -> batchnorm -> relu -> dropout -> pool
conv1, b1 = conv(xShaped, 1, 64, [5,5], name='conv1')
norm1, updateEma1 = batchNorm(conv1, tst, itera, b1, isConv=True)
relu1 = relu(norm1)
drop1 = dropout(relu1, isConv=True)
pool1 = pool(drop1, shape=[3,3], strides=[1, 3, 3, 1])

#2 conv -> batchnorm -> relu -> dropout -> pool
conv2, b2 = conv(pool1, 64, 128, [5,5], name='conv2')
norm2, updateEma2 = batchNorm(conv2, tst, itera, b2, isConv=True)
relu2 = relu(norm2)
drop2 = dropout(relu2, isConv=True)
pool2 = pool(drop2, shape=[3,3], strides=[1, 3, 3, 1])

#3 conv -> batchnorm -> relu -> dropout -> pool
conv3, b3 = conv(pool2, 128, 96, [5,5], name = 'conv3')
norm3, updateEma3 = batchNorm(conv3, tst, itera, b3, isConv=True)
relu3 = relu(norm3)
drop3 = dropout(relu3, isConv=True)
pool3 = pool(drop3, shape=[2,2], strides=[1, 2, 2, 1])

#4 flatten -> dense -> batchnorm -> relu -> dropout
pool3Flat = tf.reshape(pool3, [-1, 16 * 2 * 96])
dense1, bd1 = dense(pool3Flat, shape=[16 * 2 * 96, 1000])
normd1, updateEmad1 = batchNorm(dense1, tst, itera, bd1)
relud1 = relu(normd1)
dropd1 = dropout(relud1)

#5 dense -> batchnorm -> relu -> dropout
dense2, bd2 = dense(dropd1, shape=[1000, 100])
normd2, updateEmad2 = batchNorm(dense2, tst, itera, bd2)
relud2 = relu(normd2)
dropd2 = dropout(relud2) # seq len = 100 ?

#6 bilstm -> bilstm -> dense -> relu
lstm1 = bilstm(dropd2, 100, lstmSize, "bilstm1")
lstm2 = bilstm(lstm1, 100, lstmSize, "bilstm2")

# group exp moving average ops
ema = tf.group(updateEma1, updateEma2, updateEma3, updateEmad1, updateEmad2)

# final dense
out, _ = dense(lstm2, [100, len(charmap) + 1])
y_ = relu(out)

#7 Connectionist Temporal Classification
loss = tf.nn.ctc_loss(labels=y, inputs=y_, 100, time_major=True)
meanLoss = tf.reduce_mean(loss)

# Training
trainStep = tf.train.AdamOptimizer(lr).minimize(meanLoss)

# Evaluation
