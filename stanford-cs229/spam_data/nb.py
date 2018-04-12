import numpy as np
import sys

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    sys.stderr.write('{} rows {} cols and {} tokens\n'.format(rows, cols, len(tokens)))
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    # sys.stderr.write(' '.join( [ tokens[x] for x in range(len(tokens)) if matrix[0, x] > 0 ]) + '\n')
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    V = matrix.shape[1]
    ###################
    state = np.zeros(V + 1)
    y1 = np.sum(category)
    y0 = len(category) - y1
    state[V]  = np.log(1.0 * y1 / y0)
    
    # The total number of tokens per category.
    # Per token log-ratio
    tokens1 = np.sum(matrix[category > 0, :], 0)
    tokens0 = np.sum(matrix[category == 0, :], 0)

    numToken1 = np.sum(tokens1)
    numToken0 = np.sum(tokens0)

    state[0:V] = np.log((tokens1 + 1) / (numToken1 + V)) - np.log((tokens0 + 1) / (numToken0 + V))
    
    idx = list(range(V))
    sys.stderr.write('prio = {:.4f}\n'.format(np.exp(state[V])))
    idx.sort(key = lambda i: np.abs(state[i]), reverse=True)
    for i in range(20):
        sys.stderr.write('#{} {} = {:.4f}\n'.format(idx[i], tokenlist[idx[i]], state[idx[i]]))
    ###################
    return state

def nb_test(matrix, state):
    V = matrix.shape[1]
    output = np.zeros(matrix.shape[0])
    ###################
    output = ((matrix.dot(state[0:V]) + state[V]) > 0).astype(int)
    # print(output.shape)
    ###################
    return output

def evaluate(output, label):
    error = np.not_equal(output, label).astype(int).sum() * 1. / len(output)
    print('Error: %1.4f' % error)

for f in [
    'MATRIX.TRAIN.50',
    'MATRIX.TRAIN.100',
    'MATRIX.TRAIN.200',
    'MATRIX.TRAIN.400',
    'MATRIX.TRAIN.800',
    'MATRIX.TRAIN.1400',
    'MATRIX.TRAIN',
]:
    trainMatrix, tokenlist, trainCategory = readMatrix(f)
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
