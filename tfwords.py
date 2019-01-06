"""
Joerg Evermann, 2016

Adapted from and based upon the TensorFlow RNN tutorial provided by Google

"""

import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn 
import numpy
import time
import os
import readWords

# Large Model
#
# batchSize = 20
# numUnrollSteps = 20
# hiddenSize = 500
# dropoutProb = 0.2
# numLayers = 2
# maxGradNorm = 5
# initScale = 0.10
# numEpochsFullLR = 25
# numEpochs = 50
# baseLearningRate = 1.0
# lrDecay = 0.75

# Small Model
#
batchSize = 20
numUnrollSteps = 10
hiddenSize = 50
dropoutProb = 0.0
numLayers = 2
maxGradNorm = 5
initScale = 0.10
numEpochsFullLR = 25
numEpochs = 50
baseLearningRate = 1.0
lrDecay = 0.75

numRuns = 3

resultFile = open("resultFile.csv", 'w')

for dataset in ([
"BPI_Challenge_2013_incidents.extract.txt",
"BPI_Challenge_2013_incidents.extract.with.group.txt",
"BPI_Challenge_2013_problems.extract.txt",
"BPI_Challenge_2013_problems.extract.with.group.txt",
"BPIC_Challenge_2012.extract.complete.txt",
"BPIC_Challenge_2012.extract.complete.with.resource.txt",
"BPIC_Challenge_2012.extract.txt",
"BPIC_Challenge_2012.extract.with.resource.txt",
"BPIC_Challenge_2012_A.extract.complete.txt",
"BPIC_Challenge_2012_A.extract.complete.with.resource.txt",
"BPIC_Challenge_2012_A.extract.txt",
"BPIC_Challenge_2012_A.extract.with.resource.txt",
"BPIC_Challenge_2012_O.extract.complete.txt",
"BPIC_Challenge_2012_O.extract.complete.with.resource.txt",
"BPIC_Challenge_2012_O.extract.txt",
"BPIC_Challenge_2012_O.extract.with.resource.txt",
"BPIC_Challenge_2012_W.extract.complete.txt",
"BPIC_Challenge_2012_W.extract.complete.with.resource.txt",
"BPIC_Challenge_2012_W.extract.txt",
"BPIC_Challenge_2012_W.extract.with.resource.txt" ]):

    resultFile.write(dataset + ", ")

    for runNum in range(0, numRuns):

        tf.reset_default_graph()

        # Read the data
        trainwords, testwords, validwords, vocabSize = readWords.words_raw_data("./bpi_data", dataset)

        print("Vocbulary Size: %.0f" %(vocabSize))

        input_data = tf.placeholder(tf.int64, [batchSize, numUnrollSteps])
        targets = tf.placeholder(tf.int64, [batchSize, numUnrollSteps])

        lstm_cell = tf.nn.rnn_cell.LSTMCell(hiddenSize, forget_bias=0.0)
        # Add a propabilistic dropout to cells of the LSTM layer
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1.0-dropoutProb)
        # Replicate this (including dropout) to additional layers
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * numLayers)

        # initial state of all cells is zero
        initialState = cell.zero_state(batchSize, tf.float32)

        with tf.device("/cpu:0"):
            # embedding = tf.get_variable("embedding", [vocabSize, hiddenSize])
            embedding = tf.Variable(tf.random_uniform([vocabSize, hiddenSize], -initScale, initScale),  name="embedding")
            inputs = tf.nn.embedding_lookup(embedding, input_data)
            # Add a probabilistic dropout to inputs as well
            # inputs = tf.nn.dropout(inputs, 1.0-dropoutProb)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, numUnrollSteps, 1)]
        outputs, state = rnn.static_rnn(cell, inputs, initial_state=initialState)

        print(tf.concat(outputs, 1))
        output = tf.reshape(tf.concat(outputs,1), [-1, hiddenSize])
        # softmax_w = tf.get_variable("softmax_w", [hiddenSize, vocabSize])
        # softmax_b = tf.get_variable("softmax_b", [vocabSize])
        softmax_w = tf.Variable(tf.random_uniform([hiddenSize, vocabSize], -initScale, initScale), name="softmax_w")
        softmax_b = tf.Variable(tf.random_uniform([vocabSize],  -initScale, initScale), name="softmax_b")
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example( [logits], [tf.reshape(targets, [-1])], weights=[tf.ones([batchSize * numUnrollSteps])] )
        cost = tf.reduce_sum(loss) / batchSize
        finalState = state

        correct_prediction = tf.cast(tf.nn.in_top_k(logits, tf.reshape(targets, [-1]), 1), tf.float32)
        numCorrectPredictions = tf.reduce_sum(correct_prediction)
        accuracy = tf.reduce_mean(correct_prediction)

        oneHotTargets = tf.one_hot(targets, depth=vocabSize, on_value=1.0, off_value=0.0)
        reshapedTargets = tf.reshape(oneHotTargets, shape=(batchSize*numUnrollSteps, vocabSize))
        crossEntropy = tf.reduce_mean(-tf.reduce_sum(reshapedTargets * tf.log(tf.sigmoid(logits)), reduction_indices=[1]), reduction_indices=[0])

        learningRate = tf.Variable(0.0, trainable=False)
        trainableVars = tf.trainable_variables()
        # clip the gradients, prevent from getting too large too fast
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainableVars), maxGradNorm)
        optimizer = tf.train.GradientDescentOptimizer(learningRate)
        train_op = optimizer.apply_gradients(zip(grads, trainableVars))

        init_op = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init_op)

        lastPrecision = 0
        lastPerplexity = 0
        finalEpoch = 0
        # Training starts here
        for i in range(numEpochs):
            # Adjust the learning rate by decaying it
            # and set the appropriate variable in the graph
            sess.run(tf.assign(learningRate, baseLearningRate*lrDecay**max(i - numEpochsFullLR, 0.0)))
            # Get the learning rate and print it
            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(learningRate)))

            epochNumBatches = ((len(trainwords) // batchSize) - 1) // numUnrollSteps
            start_time = time.time()
            # accumulated cross entropy
            accumCrossEnt = 0.0
            # accumulated costs over the unroll steps
            accumCosts = 0.0
            # number of correct predictions
            accumNumCorrPred = 0
            # number of iterations/unroll steps
            iters = 0
            state_is_tuple = False
            #state = initialState.eval(session=sess)
            state = initialState.eval()
            for batchNum, (x, y) in enumerate(readWords.words_iterator(trainwords, batchSize, numUnrollSteps)):
                batchNumCorrPred, batchCrossEnt, batchCost, state, _ = sess.run([numCorrectPredictions, crossEntropy, cost, finalState, train_op], {input_data: x, targets: y, initialState: state})
                accumCosts += batchCost
                accumNumCorrPred += batchNumCorrPred
                accumCrossEnt += batchCrossEnt
                iters += numUnrollSteps
                if batchNum % (epochNumBatches // 10) == 10:
                    print("Epoch percent: %.3f perplexity: %.3f speed: %.0f wps number of correct predictions: %.0f precision: %.3f cross-entropy: %.3f" %
                          (batchNum * 1.0 / epochNumBatches, numpy.exp(accumCosts / iters), iters * batchSize / (time.time() - start_time), accumNumCorrPred, accumNumCorrPred / (iters * batchSize), accumCrossEnt / batchNum))
            thisPrecision = accumNumCorrPred / (iters * batchSize)
            thisPerplexity = numpy.exp(accumCosts / iters)
            print("Epoch summary: perplexity: %.3f speed: %.0f wps number of correct predictions: %.0f precision: %.3f cross-entropy: %.3f" %
                  (numpy.exp(accumCosts / iters), iters * batchSize / (time.time() - start_time), accumNumCorrPred, thisPrecision, accumCrossEnt / batchNum))
            finalEpoch=i
            if (thisPrecision < lastPrecision):
                break
            else:
                lastPrecision = thisPrecision
                lastPerplexity = thisPerplexity

        resultFile.write("%.0f, %.3f, %.3f, " % (finalEpoch, lastPrecision, lastPerplexity) )

    resultFile.write("\n")
    resultFile.flush()
    os.fsync(resultFile.fileno())

"""
print("Training done. Begin testing.")
# Testing starts here. Treat as regular bunch of batchs, only difference is no optimization operation is called!
epochNumBatches = ((len(testwords) // batchSize) - 1) // numUnrollSteps
start_time = time.time()
# accumulated costs over the unroll steps
accumCosts = 0.0
# number of correct predictions
accumNumCorrPred = 0
# number of iterations/unroll steps
iters = 0
state = initialState.eval(session=sess)
for batchNum, (x, y) in enumerate(readWords.words_iterator(testwords, batchSize, numUnrollSteps)):
    batchNumCorrPred, batchCost, state, _ = sess.run([numCorrectPredictions, cost, finalState, tf.no_op()],
                                                     {input_data: x, targets: y, initialState: state})
    accumCosts += batchCost
    accumNumCorrPred += batchNumCorrPred
    iters += numUnrollSteps
    if batchNum % (epochNumBatches // 10) == 10:
        print(
        "Testing percent: %.3f perplexity: %.3f speed: %.0f wps number of correct predictions: %.0f precision: %.3f" %
        (batchNum * 1.0 / epochNumBatches, np.exp(accumCosts / iters), iters * batchSize / (time.time() - start_time), accumNumCorrPred, accumNumCorrPred / (iters * batchSize)))
print("Testing summary: perplexity: %.3f speed: %.0f wps number of correct predictions: %.0f precision: %.3f" %
      (np.exp(accumCosts / iters), iters * batchSize / (time.time() - start_time), accumNumCorrPred, accumNumCorrPred / (iters * batchSize)))
"""

resultFile.close()