import tensorflow as tf
import numpy as np
import pandas as pd
import sys


#Builds a network inner layer, to be initialized using Xavier initialization
#prevLayer: the output tensor of the previous layer
#inSize, outSize: the size of the previous layer and this one; used to create the weights matrix
#wName: a name for the weights matrix
#activation: reference to the Tensorflow activation function
#outName: a name to be assigned to the output tensor
def buildLayer(prevLayer, inSize, outSize, wName, activation = tf.nn.tanh, outName = None):
    #Build the weights matrix, set initialization to Xavier method
    W = tf.get_variable(wName, shape=[inSize, outSize], initializer=tf.contrib.layers.xavier_initializer())

    #Add the weights to the regularization set
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)

    #Build the bias variable
    b = tf.Variable(tf.ones([outSize], dtype=np.float32))

    #Create and return the output tensor as activation(weighed sum of previous layer)
    return activation(tf.matmul(prevLayer, W) + b, name=outName)

def loadNetwork(path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(path + '.meta')
    saver.restore(sess, path)
    graph = sess.graph
    x = graph.get_tensor_by_name("input:0")
    y = graph.get_tensor_by_name("output:0")
    top10 = tf.nn.top_k(y, 10, sorted=False, name="top10")
    #top10 = graph.get_tensor_by_name("top10:0")
    return [sess, x, y, top10]

#Creates a neural network with 1 or 2 inner layers + dropout on each and regularization
#tailored for recommending top 10 books for each user
#Uses Adam as training algorithm
#Takes 2 datasets: training and testing and periodically calculates the loss on both and the accuracy(hit rate) on the test
#Can save the network as well as a file containing the training evolution as CSV
#All test and train sets are matrices of row vectors
#trainX: the training input (user features)
#trainY: the training output (user/book scores)
#testX: the testing input (user features)
#testXUnseen: the subset of indices from the testing input that has not been used in calculating the book score. Used as a
# more reliable measure of the hitrate
#testY: the testing output (real user selection)
#epochs: number of epochs to train the network
#outputEachStep: will display the losses and accuracy each N steps (+output to file if save_to is defined)
#layer1Size: size of the first inner layer
#layer2Size: size of the second inner layer
#keepProb: keep probability for the dropout layers (common for both)
#reg: regularization parameter
#learnRate: the learning rate
#minLoss: training will stop when the loss falls below this threshold
#saveTo: if defined, the network will be saved under this path/filename. Also a file containing the training evolution
#will be saved in path/name_progress
def buildScoreBasedNetwork(trainX, trainY, testX, testXUnseen, testY, epochs=20000, outputEachStep=1000, layer1Size=10,
                           layer2Size = None, keepProbValue = 1, reg = 0.1, learnRate = 1E-4, minLoss = 10, saveTo=None
                           ):
    
    #The input and output sizes
    inputSize = trainX.shape[1]
    outputSize = trainY.shape[1]

    #Create a graph for this network configuration
    graph = tf.Graph()

    #Create the network configuration
    with graph.as_default():
        #Placeholder for the dropout keep probabilities, with default set to 1
        keepProb = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        #The input tensor
        x = tf.placeholder(tf.float32, shape=[None, inputSize], name="input")

        #Placeholder for the dataset output
        y_ = tf.placeholder(tf.float32, shape=[None, outputSize])

        #Layer 1
        layer1 = buildLayer(x, inputSize, layer1Size, "W1")

        #Dropout for layer 1
        layer1do = tf.nn.dropout(layer1, keep_prob = keepProb)

        if layer2Size is not None:
            #Layer 2
            layer2 = buildLayer(layer1do, layer1Size, layer2Size, "W2")

            #Dropout for layer 2
            layer2do = tf.nn.dropout(layer2, keep_prob=keepProb)

            #Output layer
            y = buildLayer(layer2do, layer2Size, outputSize, "W3", activation=tf.identity, outName="output")
        else:
            #Output layer
            y = buildLayer(layer1do, layer1Size, outputSize, "W3", activation=tf.identity, outName="output")

        #The cost is the sum of squared differences
        loss = tf.reduce_sum(tf.squared_difference(y, y_))

        #This gets the largest 10 values and their indices on each data point
        top10 = tf.nn.top_k(y, 10, sorted=False, name="top10")

        # Define the regularization term and add it to the cost function
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg)
        regVariables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
        loss += regTerm

        #Train via ADAM
        train_step = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)

        #Create the session and run the global initializer
        sess = tf.Session(graph = graph)
        sess.run(tf.global_variables_initializer())


    #If set to save the network, will also save a progress file showing the loss and hit rate evolution during training
    if saveTo is not None:

        #The output file will have the same name as the model, suffixed by "_progress"
        saveProgress = saveTo + "_progress.txt"

        #Open the file and add the header
        progressFile = open(saveProgress, mode="w")
        progressFile.write("TRAIN_LOSS,TEST_HR_ALL,TEST_HR_UNSEEN\n")
    else:
        progressFile = None


    #Accumulators for the return values
    hrAll = 0
    hrUnseen = 0
    trainLoss = 0

    #Train the network for the specified number of epochs
    for i in range(epochs):

        #As defined by outputEachStep, will calculate cost and accuracy, print it on screen and write in the output file (if it exists)
        if i % outputEachStep == 0 or i == epochs - 1:

            #Get the loss of the training set
            [trainLoss] = sess.run([loss], feed_dict={x: trainX, y_: trainY})

            #Calculate and display the overall and test hit rate
            [hrAll, hrUnseen] = testTop10HitRate(sess, x, y_, top10, testX, testY, testXUnseen)
            print("Epoch {:d} | Train loss={:.4f} | Test HR all/unseen={:.4f}/{:.4f}".format(i + 1, trainLoss, hrAll, hrUnseen))

            #Write to file, if it exists
            if progressFile is not None:
                progressFile.write("{:.4f},{:.4f},{:.4f}\n".format(trainLoss, hrAll, hrUnseen))

        #If fallen below the minimum loss, stop training and return
        if trainLoss > 0 and trainLoss < minLoss:
            break

        #Run a training epoch, enabling dropout
        sess.run(train_step, feed_dict={x: trainX, y_: trainY, keepProb: keepProbValue})

    #If set to save the network, do it. Also close the progress file
    if saveTo is not None:
        if progressFile is not None:
            progressFile.close()
        saver = tf.train.Saver()
        saver.save(sess, saveTo)

    return [trainLoss, hrAll, hrUnseen, sess, x, y, y_, top10]

#Given a network and a test set, calculates the hit rate
#Can highlight the hit rate on a set of indices (used when testing the entire dataset)
#sess: the TF session
#x: the input tensor
#y: the output tensor
#y_: the test data placeholder
#top10: TF operation to get the top 10 values
#testX: the input dataset
#testY: the output dataset
#highlightIndices: np array of indices to calculate the hit rate separately
def testTop10HitRate(sess, x, y_, top10, testX, testY, highlightIndices = np.asarray([])):
    # Get the loss of the testing set and the indices of maximum values
    [indices] = sess.run([top10.indices], feed_dict={x: testX})

    # The following sequence calculates the hitrate for the top 10 values.
    # It's quite ugly as it needs to loop through the dataset row by row
    # It appears that TF has no way of doing this in one line
    hrAll = 0
    hrUnseen = 0
    # For each line in the test set
    for j in range(testY.shape[0]):
        # Calculate the sum of the corresponding indices (10 = full hit, 0 = no hit)
        sum = testY[j, indices[j]].sum()
        hrAll += sum
        if j in highlightIndices:
            hrUnseen += sum
    # Divide to get the normalized hitrate
    hrAll /= testY.shape[0] * 10
    if highlightIndices.shape[0] != 0:
        hrUnseen /= highlightIndices.shape[0] * 10

    return [hrAll, hrUnseen]

#Creates a dataset with inferred book scores baset on the given 1/0 dataset
#The test indices do not take part in calculating the score, however they are assigned the final score in order to
# test the reference model
#dataset: the user traits + user/book dataset, as returned by createDataset(...)
#testIndices: numpy array of indices from the dataset to be used for testing
#returns: a list [train loss, global hit rate, test hit rate, session, x, y, top10] obtained from training the network
def buildAndTestScoreBasedModel(dataset, testIndices):
    #Split the given dataset into test and train, according to the given indices
    testDS = dataset.iloc[testIndices, :]
    trainDS = dataset.loc[~dataset.index.isin(testDS.index)]

    #Create a copy of the original dataset
    joinDFAdjusted = dataset.copy(deep=True)

    #Add to each user/book cell an initial score equal to the mean of the book's selection
    # (i.e. how many times was it selected in the training dataset)
    #means = trainDS.iloc[:, 10:].mean()
    joinDFAdjusted.iloc[:, 10:] = 0
    #joinDFAdjusted.iloc[:, 10:] = joinDFAdjusted.iloc[:,10:].add(means)
    #joinDFAdjusted.iloc[testIndices, 10:] = 0

    #For each user trait
    for i in range(10):
        #Calculate the mean selection for each book from users having this trait, using the training dataset
        means = trainDS[trainDS.iloc[:, i] == 1].iloc[10:, :].mean()
        #The above also calculates the mean of user traits, reset this to 0 (seems like assigning on double indexing
        # in pandas is tricky)
        means.iloc[0:10] = 0

        #Update the new dataset by adding the corresponding mean to each user/book cell
        joinDFAdjusted[joinDFAdjusted.iloc[:, i] == 1] = joinDFAdjusted[joinDFAdjusted.iloc[:, i] == 1].add(means)

        #This is just commented code from previous attempts
        '''for j in range(i+1, 10):
            means = trainDS[trainDS.iloc[:, i] + trainDS.iloc[:, j] == 2].iloc[10:, :].mean()
            means.iloc[0:10] = 0
            joinDFAdjusted[joinDFAdjusted.iloc[:, i] + joinDFAdjusted.iloc[:, j] == 2] = \
                joinDFAdjusted[joinDFAdjusted.iloc[:, i]  + joinDFAdjusted.iloc[:, j] == 2].add(means)
            for k in range(j + 1, 10):
                means = trainDS[trainDS.iloc[:, i] + trainDS.iloc[:, j]  + trainDS.iloc[:, k] == 3].iloc[10:, :].mean().fillna(value=0)
                means.iloc[0:10] = 0
                joinDFAdjusted[joinDFAdjusted.iloc[:, i] + joinDFAdjusted.iloc[:, j] + joinDFAdjusted.iloc[:, k] == 3] = joinDFAdjusted[
                    joinDFAdjusted.iloc[:, i] + joinDFAdjusted.iloc[:, j] + joinDFAdjusted.iloc[:, k] == 3].add(means)
        '''
    #Normalize the dataset (user/book cells only) to 0 mean and 1 std deviation (recommended for neural networks datasets)
    joinDFAdjusted.iloc[:, 10:] = (joinDFAdjusted.iloc[:, 10:] - joinDFAdjusted.iloc[:, 10:].mean()) / joinDFAdjusted.iloc[:, 10:].std()

    #Test the hit rate of top 10 recommendations on the entire dataset + test dataset based on the score calculated above
    testHR = 0
    allHR = 0
    #Parse each row in the new dataset (storing the score)
    for i in range(joinDFAdjusted.shape[0]):
        #We only care about the user/book columns, available from position 10 on
        series = joinDFAdjusted.iloc[i, 10:]

        #Get the top 10 score values + their indices
        largest = series.nlargest(10)

        #Slice the original dataset by the row name + top 10 largest indices and sum the values
        #10 would mean full hit, 0 full miss
        sum = dataset.loc[series.name, largest.index].sum()

        #Update the overall and test hit rate accumulators
        allHR += sum
        if i in testIndices:
            testHR += sum

    #Display the hit rates
    print("Base HR on all/test: {:.4f} / {:.4f}".format(allHR / dataset.shape[0], testHR / testIndices.shape[0]))

    #Now create the datasets for training and testing the neural network
    #The testing dataset is the entire original dataset, with testIndices to highlight the hit rate on the unused data
    #The training dataset = the score based dataset without the test indices
    trainDS = joinDFAdjusted.loc[~dataset.index.isin(testDS.index)]

    #Build and train the network, returning the train loss and hit rates
    return buildScoreBasedNetwork(trainDS.iloc[:, 0:10].as_matrix(), trainDS.iloc[:, 10:].as_matrix(),
                           dataset.iloc[:, 0:10].as_matrix(), testIndices,
                           dataset.iloc[:, 10:].as_matrix(), saveTo=None)

#Performs a K-Fold Cross-validation on the dataset
#Was used to look for a good combination of network parameters
#Only minimal error checking is performed
#dataset = the user traits + user/book dataset as returned by createDataset(...)
def crossValidate(entireDataset, folds = 10):
    dataset = entireDataset.sample(frac = 0.9)
    validationDs = entireDataset.drop(dataset.index)

    #Number of rows in dataset
    modelSize = dataset.shape[0]

    #Size of a fold, also check if valid
    foldSize = int(modelSize / folds)
    if foldSize < 1:
        raise ValueError("Adjust the fold size (model size is {:d}, total folds is {:d})".format(modelSize, foldSize))

    #This will store the loss, global and test hit rate
    [globalLoss, globalHrAll, globalHrUnseen] = [0, 0, 0]

    sess = None
    bestHrUnseen = 0

    #For each fold
    for fold in range(folds):

        #Determine the fold's start and end index, create a numpy array with the indices
        fromIndex = fold * foldSize
        toIndex = min(fromIndex + foldSize + 1, modelSize)
        testIndices = np.arange(fromIndex, toIndex)

        #Create and test the model
        #res is a list with the same structure as results
        [crtLoss, crtHrAll, crtHrUnseen, crtSess, crtX, crtY, crtY_, crtTop10] = buildAndTestScoreBasedModel(dataset, testIndices)

        #Update the results
        [globalLoss, globalHrAll, globalHrUnseen] = [globalLoss + crtLoss, globalHrAll + crtHrAll, globalHrUnseen + crtHrUnseen]
        print("Testing {:d} - {:d}, loss/HR-all/HR-unseen: {:.2f}/{:.2f}/{:.2f}".format(fromIndex, toIndex, crtLoss, crtHrAll, crtHrUnseen))

        if bestHrUnseen < crtHrUnseen:
            if sess is not None:
                sess.close()
            [bestHrUnseen, sess, x, y, y_, top10] = [crtHrUnseen, crtSess, crtX, crtY, crtY_, crtTop10]
            print("Selected as best configuration")
        else:
            crtSess.close()

    #Print the global results
    print("Total loss/HR-all/HR-unseen: {:.2f}/{:.2f}/{:.2f}".format(globalLoss/folds, globalHrAll/folds, globalHrUnseen/folds))

    if sess is not None:
        [hrValidation, _] = testTop10HitRate(sess, x, y_, top10, validationDs.iloc[:, :10].as_matrix(), validationDs.iloc[:, 10:].as_matrix())

        print("Hit rate for selected model on the validation set: {:.4f}".format(hrValidation))
        choice = input("Save the network? (Y = yes/any = no): ")
        if choice == "Y":
            with sess.graph.as_default():
                saver = tf.train.Saver()
                saver.save(sess, "nn/good1")
        sess.close()

def loadAndTestNetwork():
    pass

#Loads 2 files, one containing the users database the other the user/book preferences
#No error checking is performed, the files must exist and have the same structure as user_char.csv and user_book.csv
#usersFName: name of file containing the users
#preferencesFName: name of file containing the user/book preferences
#return: a joined dataset containing each user and his book selection on rows
# (the traits are on the first 10 columns, book selection on the next 1000)
def createDataset(usersFName, preferencesFName):
    # Read the users/features dataset; use explicit column range because it gets confused by the last comma
    usersDF = pd.read_csv(usersFName, usecols=range(11), index_col=0)
    usersDF = usersDF.astype(np.float32, copy=False)

    # Read the users/books dataset; use explicit column range because it gets confused by the last comma
    preferencesDF = pd.read_csv(preferencesFName, usecols=range(1001), index_col=0)
    preferencesDF = preferencesDF.astype(np.float32, copy=False)

    #Join and return the 2 datasets into a single one on the index column (username)
    #Inner join - removes records not present in both datasets
    return usersDF.join(preferencesDF, how='inner')

#Loads and tests the network on a given dataset
def loadAndTestNetwork(dataset):
    [sess, x, y, top10] = loadNetwork("nn/good1")
    [hr, _] = testTop10HitRate(sess, x, None, top10, dataset.iloc[:, :10].as_matrix(), dataset.iloc[:, 10:].as_matrix())
    print("Hit rate: {}".format(hr))

#Used to test on test_user_book.csv
def testUserBook():
    dataset = createDataset("user_char.csv", "test_user_book.csv")
    loadAndTestNetwork(dataset)

#Used to test on user_book.csv
def testOriginal():
    dataset = createDataset("user_char.csv", "user_book.csv")
    loadAndTestNetwork(dataset)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "check", "test"):
        print("Usage: python tf_rec.py (build|check|test)")
    else:
        if sys.argv[1] == "build":
            dataset = createDataset("user_char.csv", "user_book.csv")
            crossValidate(dataset, folds=10)
        elif sys.argv[1] == "check":
            testOriginal()
        elif sys.argv[1] == "test":
            testUserBook()

