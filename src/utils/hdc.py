import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from typing import List

# hv generation

def gen_hv(seed,D=10000):
    rng = np.random.default_rng(seed)
    hv = rng.choice([-1, 1], size=D).astype(np.int16)
    return hv

def gen_im(seed,D=10000,N=64):
    rng = np.random.default_rng(seed)
    im = []
    for n in range(N):
        im.append(gen_hv(rng,D))
    im = np.vstack(im)
    return im

def gen_sim_im(seed,D=10000,N=64,dist=0.5):
    rng = np.random.default_rng(seed)
    numFlip = round(0.5*(1 - (1 - 2*dist)**0.5)*D)
    AM = np.zeros((N,D),dtype=np.int16)
    seedHV = rng.choice([-1,1],size=D).astype(np.int16)
    for i in range(N):
        flipIdx = rng.permutation(D)[:numFlip]
        flipBits = np.ones(D)
        flipBits[flipIdx] = -1
        AM[i,:] = seedHV*flipBits
    return AM
    

def gen_cim(seed,D=10000,N=16,span=5000):
    rng = np.random.default_rng(seed)
    cim = np.ones((N,D)).astype(np.int)
    flipBits = rng.permutation(D)[:int(D/2)]
    cim[:,flipBits] = -1

    flipBits = rng.permutation(D)[:span].astype('int')
    flipAmt = np.round(np.linspace(0,span,N)).astype('int')
    for i in range(N):
        cim[i,flipBits[:flipAmt[i]]] *= -1
    return cim


# hv manipulation

def bipolarize(seed,im):
    rng = np.random.default_rng(seed)
    X = np.copy(im)
    X[X > 0] = 1
    X[X < 0] = -1
    X[X == 0] = gen_hv(rng,len(X[X == 0]))
    return X.astype(np.int16)
    
def centroids(seed,X,label=None):
    rng = np.random.default_rng(seed)
    counts=[]
    if label is not None:
        cLabel = np.unique(label)
        c = np.zeros((len(cLabel), X.shape[1]))
        cint = np.zeros((len(cLabel), X.shape[1]))
        for i,l in enumerate(cLabel):
            c[i,:] = bipolarize(rng,np.sum(X[label==l],axis=0))
            cint[i,:] = np.sum(X[label==l],axis=0)
            counts.append(label==l)
    else:
        c = bipolarize(rng,np.sum(X,axis=0)).reshape(1,-1)
        cLabel = [0]
    return cLabel, c.astype(np.int16),cint, counts

def bin_hvs(X,D,n_bins=10):
    dmax=0.5
    step=1/(n_bins-1)
    levels=step*[1,2,3,4,5,6,7,8]
    distances=cdist([X[0]],X,'hamming')

def spatial_encode(seed,feat,im):
    rng = np.random.default_rng(seed)
    spatialHV = np.zeros((len(feat),im.shape[1])).astype(np.int16)
    for i,f in enumerate(feat):
        spatialHV[i] = bipolarize(rng,np.sum(f.reshape(-1,1)*im, axis=0))
    return spatialHV

def temporal_encode(hv, N=5):
    tempHV = np.ones((len(hv),hv.shape[1])).astype(np.int16)
    for i in range(N-1,len(hv)):
        for n in range(N):
            tempHV[i] *= np.roll(hv[i-n],n)
    return tempHV

def sensor_encode(emg_hv: np.ndarray, acc_hv: np.ndarray, force_hv: np.ndarray) -> np.ndarray:
    return hd_mul(hd_mul(emg_hv, acc_hv), force_hv)

def project_gesture_to_hv(gestureLabel: np.int16):
    '''
    * uses the gesture label as the seed for an rng,
    * pulls D random bits from that rng. Should be the same
    * projection each time
    '''
    return gen_hv(gestureLabel)

def project_actuation_to_hv(actuationLabel: np.int16):
    return gen_hv(actuationLabel)

def bind_gesture_actuator(gesture_hv: np.ndarray, actuator_hv: np.ndarray):
    return hd_mul(gesture_hv, actuator_hv)

def bundle_gesture_actuator_pairs(hvs: List[np.ndarray]):
    return hd_threshold(np.sum(hvs, axis=0))

def extract_gesture_actuator_pair_from_pv(pv:np.ndarray, gesture:np.ndarray, actuator:np.ndarray = None):
    #if gesture != None:
    return hd_mul(pv, gesture)
    # if actuator != None:
    #     return hd_mul(pv, actuator)
    # else :
    #     Exception("One of gesture or actuator must not be None")

def hd_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Return element-wise multiplication between bipolar HD vectors
    # inputs:
    #   - A: bipolar HD vector
    #   - B: bipolar HD vector
    # outputs:
    #   - A*B: bipolar HD vector
    return np.multiply(A,B,dtype = np.int8)

def hd_perm(A: np.ndarray) -> np.ndarray:
    # Return right cyclic shift of input bipolar vector
    # inputs:
    #   - A: bipiolar HD vector
    # outputs:
    #   - rho(A): bipolar HD vector
    return np.roll(A,1)

def hd_threshold(A):
    # Given integer vector, threshold at zero to bipolar
    # inputs:
    #   - A: bipolar HD vector
    # outputs:
    #   - [A]: bipolar HD vector
    #return (np.greater_equal(A,0, dtype=np.int8)*2-1)
    print(np.shape(A))
    return (np.greater_equal(A,0)*2-1)

# hv classification

def distances(v1, v2, metric='hamming'):
    return cdist(v1, v2, metric)

def dotprod(v1,v2):
    return sum([n1*n2 for n1,n2 in zip(list(v1),list(v2))])

def classify(v,am,metric,amLabels=None):
    d = distances(v,am,metric)
    if amLabels is not None:
        label = amLabels[np.argmin(d,axis=1)]
    else:
        label = np.argmin(d,axis=1)
    return label

def cross_val(seed,X,y,g,n_splits=10):
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    acc = []
    for trainIdx, testIdx in skf.split(X,g):
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        AM = centroids(rng,XTrain,label=yTrain)
        pred = classify(XTest,AM[1],'hamming',amLabels=AM[0])
        acc.append(sum(pred == yTest)/len(pred))
    return np.mean(acc)

def cross_val_multi(seed,X,y,g,gLabel,n_splits=10):
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    acc = []
    for trainIdx, testIdx in skf.split(X,g):
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        gTrain, gTest = g[trainIdx], g[testIdx]
        AM = centroids(rng,XTrain,label=gTrain)
        pred = classify(XTest,AM[1],'hamming',amLabels=gLabel[AM[0]])
        acc.append(sum(pred == yTest)/len(pred))
    return np.mean(acc)

def cross_val_tuples(seed,X,y,n_splits=10):
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    acc = []
    distances = []
    for trainIdx, testIdx in skf.split(X,y):
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        AM = centroids(rng,XTrain,label=yTrain)
        pred, dists = classify(XTest,AM[1],'cosine',amLabels=AM[0])
        acc.append(sum(pred == yTest)/len(pred))
        distances.append(dists)
    return np.mean(acc), distances

def train_test_split_accuracy(seed,XTrain,XTest,yTrain,yTest):
    rng = np.random.default_rng(seed)
    AM = centroids(rng,XTrain,label=yTrain)
    pred = classify(XTest,AM[1],'hamming',amLabels=AM[0])
    acc=sum(pred == yTest)/len(pred)
    return acc,pred

# accelerometer specific

def convert_to_idx(feature,span,levels):
    idx = np.round((feature - span[0])/(span[1] - span[0])*(levels - 1)).astype('int')
    idx[idx < 0] = 0
    idx[idx > levels-1] = levels - 1
    return idx

def classify_positions(accFeat,posLabel,gestLabel,numSplit):
    skf = StratifiedKFold(n_splits=numSplit, shuffle=True)
    _, groupGP = np.unique(np.column_stack((gestLabel,posLabel)),axis=0,return_inverse=True)
    
    outLabel = -np.ones(posLabel.shape)
    for trainIdx, testIdx in skf.split(accFeat, groupGP):
        XTrain, XTest = accFeat[trainIdx], accFeat[testIdx]
        yTrain, yTest = posLabel[trainIdx], posLabel[testIdx]
        svm = SVC(decision_function_shape='ovr',kernel='linear',C=1e20)
        svm.fit(XTrain,yTrain)
        outLabel[testIdx] = svm.predict(XTest)
    
    return outLabel

#Utils

