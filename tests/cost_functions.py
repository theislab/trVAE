from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

IntType = 'int32'
FloatType = 'float32'


# calculate the squared distance between x and y
def squaredDistance(X, Y):
    # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
    # broadcasted subtraction, a square, and a sum.
    r = K.expand_dims(X, axis=1)
    return K.sum(K.square(r - Y), axis=-1)


class MMD:
    MMDTargetTrain = None
    MMDTargetTrainSize = None
    MMDTargetValidation = None
    MMDTargetValidationSize = None
    MMDTargetSampleSize = None
    kernel = None
    scales = None
    weights = None

    def __init__(self,
                 MMDLayer,
                 MMDTargetTrain,
                 MMDTargetValidation_split=0.1,
                 MMDTargetSampleSize=1000,
                 n_neighbors=25,
                 scales=None,
                 weights=None):
        if scales == None:
            print("setting scales using KNN")

        med = np.zeros(20)
        for ii in range(1, 20):
            sample = MMDTargetTrain[np.random.randint(MMDTargetTrain.shape[0], size=MMDTargetSampleSize), :]
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
            distances, dummy = nbrs.kneighbors(sample)
            # nearest neighbor is the point so we need to exclude it
            med[ii] = np.median(distances[:, 1:n_neighbors])
        med = np.median(med)
        scales = [med / 2, med, med * 2]  # CyTOF
        print(scales)
        scales = K.variable(value=np.asarray(scales))
        if weights == None:
            print("setting all scale weights to 1")
        weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        self.MMDLayer = MMDLayer
        MMDTargetTrain, MMDTargetValidation = train_test_split(MMDTargetTrain, test_size=MMDTargetValidation_split,
                                                               random_state=42)
        self.MMDTargetTrain = K.variable(value=MMDTargetTrain)
        self.MMDTargetTrainSize = K.eval(K.shape(self.MMDTargetTrain)[0])
        self.MMDTargetValidation = K.variable(value=MMDTargetValidation)
        self.MMDTargetValidationSize = K.eval(K.shape(self.MMDTargetValidation)[0])
        self.MMDTargetSampleSize = MMDTargetSampleSize
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights

    # calculate the raphy kernel applied to all entries in a pairwise distance matrix
    def RaphyKernel(self, X, Y):
        # expand dist to a 1xnxm tensor where the 1 is broadcastable
        sQdist = K.expand_dims(squaredDistance(X, Y), 0)

        # expand scales into a px1x1 tensor so we can do an element wise exponential
        self.scales = K.expand_dims(K.expand_dims(self.scales, -1), -1)
        # expand scales into a px1x1 tensor so we can do an element wise exponential
        self.weights = K.expand_dims(K.expand_dims(self.weights, -1), -1)
        # calculated the kernal for each scale weight on the distance matrix and sum them up
        return K.sum(self.weights * K.exp(-sQdist / (K.pow(self.scales, 2))), 0)

    # Calculate the MMD cost
    def cost(self, source, target):
        # calculate the 3 MMD terms
        xx = self.kernel(source, source)
        xy = self.kernel(source, target)
        yy = self.kernel(target, target)

        # calculate the bias MMD estimater (cannot be less than 0)
        MMD = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
        # return the square root of the MMD because it optimizes better
        return K.sqrt(MMD);

    def KerasCost(self, y_true, y_pred):
        # create a random subsampling of the target instances for the test set
        # This is rarely going to hit the last entry
        sample = K.cast(K.round(K.random_uniform_variable(shape=tuple([self.MMDTargetSampleSize]), low=0,
                                                          high=self.MMDTargetTrainSize - 1)), IntType)
        # this is a subset operation (not a very pretty way to do it)
        MMDTargetSampleTrain = K.gather(self.MMDTargetTrain, sample)
        # do the same for the validation set
        sample = K.cast(K.round(K.random_uniform_variable(shape=tuple([self.MMDTargetSampleSize]), low=0,
                                                          high=self.MMDTargetValidationSize - 1)), IntType)
        # and the subset operation
        MMDTargetSampleValidation = K.gather(self.MMDTargetValidation, sample)
        # create the sample based on whether we are in training or validation steps
        MMDtargetSample = K.in_train_phase(MMDTargetSampleTrain, MMDTargetSampleValidation)
        # return the MMD cost for this subset
        ret = self.cost(self.MMDLayer, MMDtargetSample)
        # pretty dumb but y_treu has to be in the cost for keras to not barf when cleaning up
        ret = ret + 0 * K.sum(y_pred) + 0 * K.sum(y_true)

        return ret
