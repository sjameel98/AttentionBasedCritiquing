import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import random
import matplotlib.pyplot as plt
from scipy.sparse import save_npz, load_npz
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD, NMF
import pandas as pd
from scipy.special import softmax

np.random.seed(seed=1)

idx2item = json.load(open("./idx_2_itemId.json"))
idx2keyphrase = json.load(open("./idx_2_keyphrase.json"))
word2idx = json.load(open("./keyphrase2idx.json"))


Rikz = load_npz("./Rtrain_item_keyphrase.npz").tocsr().toarray()
Riu = load_npz("./Rtrain.npz").tocsr().T

'''
Rikz = np.load("MovieLensItemKeyphrase.npy")
Riu = np.load("MovieLensItemUser.npy")
'''

print("Generating user-item decomposition and TCAV embeddings")

svd = TruncatedSVD(n_components=50)
U = svd.fit_transform(Riu)
Sigma = np.diag(svd.singular_values_)
V = svd.components_.T

items = U @ np.power(Sigma, 1 / 2)
users = (np.power(Sigma, 1 / 2) @ V.T).T

'''
model = NMF(n_components=50, random_state=1, max_iter=500)
items = model.fit_transform(Riu)
users = model.components_.T
'''

numKeyphrases = Rikz.shape[1]
keyphrases = np.zeros((numKeyphrases, users.shape[1]))

print("Item matrix shape: ", items.shape)
print("User matrix shape: ", users.shape)

for i in range(numKeyphrases):
    #labels = Rikz[:, i].toarray().squeeze()
    labels = Rikz[:,i].squeeze()
    clf = LinearRegression()
    clf.fit(X=items, y=labels)
    keyphrases[i, :] = clf.coef_

print("Keyphrase matrix shape: ", keyphrases.shape)
print("Finished user-item decomposition and TCAV embeddings")

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

keyphrases = (keyphrases.T / np.linalg.norm(keyphrases, axis=1).T).T
users = (users.T / np.linalg.norm(users, axis=1).T).T

IUAct = np.matmul(items, users.T)
IKAct = np.matmul(items, keyphrases.T)

sortedUI = np.argsort(-IUAct, axis=0).T
sortedIK = np.argsort(-IKAct, axis=1)

numUsers = sortedUI.shape[0]
numItems = sortedIK.shape[0]
vector_size = keyphrases.shape[1]



def generate_dataset(numCritiques, numReplicas=3, min_rank=10, max_rank=20, randomize=False):
    trials = []

    if ~randomize:
        for i in range(numUsers):
            for j in range(numReplicas):
                index = np.random.randint(low=min_rank, high=max_rank)
                itemIndex = sortedUI[i, index]
                keyphrases = []
                while len(keyphrases) < numCritiques:
                    indexC = np.random.randint(low=0, high=5)
                    keyphraseIndex = sortedIK[itemIndex, indexC]
                    if keyphraseIndex not in keyphrases:
                        keyphrases.append(keyphraseIndex)

                trial = {}
                trial["user"] = i
                trial["item"] = itemIndex
                trial["critiques"] = keyphrases
                trials.append(trial)
    else:
        for i in range(numUsers):
            replicas = np.random.randint(low=1, high=numReplicas)
            for j in range(replicas):
                index = np.random.randint(low=min_rank, high=max_rank)
                itemIndex = sortedUI[i, index]
                keyphrases = []
                sessionLength = np.random.randint(low=1, high=numCritiques)
                while len(keyphrases) < sessionLength:
                    indexC = np.random.randint(low=0, high=5)
                    keyphraseIndex = sortedIK[itemIndex, indexC]
                    if keyphraseIndex not in keyphrases:
                        keyphrases.append(keyphraseIndex)

                trial = {}
                trial["user"] = i
                trial["item"] = itemIndex
                trial["critiques"] = keyphrases
                trials.append(trial)


    return trials

def datum2inputs(datum):
    userVector = users[datum['user'], :]
    critiques = datum['critiques']

    allKeys = datum.keys()
    if (('inputs' in allKeys) and ('target' in allKeys)):
        # print("preexisting inputs")
        return datum['inputs'], datum['target']

    inputs = [torch.Tensor(userVector).to(dev)]
    for critique in critiques:
        cvector = torch.from_numpy(keyphrases[critique, :]).float().to(dev)
        inputs.append(cvector)
    target = torch.from_numpy(items[datum['item'], :]).float().to(dev)

    return inputs, target


def negative_random_samples(datum, total_samples=1):
    additionalSamples = set()
    while total_samples > 0:
        total_samples -= 1
        index = random.randint(0, 4996)
        if ((index != datum['item'])):
            additionalSamples.add(index)
    additionalSamples = list(additionalSamples)
    torchASamples = [torch.from_numpy(items[sp, :]).unsqueeze(0).to(dev).float() for sp in additionalSamples]

    return torchASamples

def negative_contrastive_samples(datum, output, total_samples=1):
    np_output = output.cpu().detach().numpy()

    results = -1 * items @ np_output
    rankings = np.argsort(results)
    listedRankings = rankings.tolist()[0:total_samples+1]
    targetInd = datum["item"]

    if targetInd in listedRankings:
        listedRankings.remove(targetInd)
    else:
        listedRankings = listedRankings[0:total_samples]

    torchASamples = [torch.from_numpy(items[sp, :]).unsqueeze(0).to(dev).float() for sp in listedRankings]
    return torchASamples

def negative_temperature_samples(datum, output, total_samples=1, temperature = 1):
    np_output = output.cpu().detach().numpy()

    results = items @ np_output
    positive_results = (results-np.min(results))**temperature
    probabilities = positive_results/positive_results.sum()

    samples = np.random.choice(items.shape[0], total_samples+1, replace=False, p=probabilities).tolist()
    targetInd = datum["item"]

    if targetInd in samples:
        samples.remove(targetInd)
    else:
        samples = samples[0:total_samples]

    torchASamples = [torch.from_numpy(items[sp, :]).unsqueeze(0).to(dev).float() for sp in samples]
    return torchASamples

def stacker(inputs, prepend=False, append=False):
    finalInputs = [inp.reshape(1, 1, 50) for inp in inputs]
    if (prepend):
        while (len(inputs) < 4):
            inputs.insert(0, torch.Tensor(50))
    elif (append):
        while (len(inputs) < 4):
            inputs.append(torch.Tensor(50))
    finalInputs = torch.cat(finalInputs, dim=1)
    return finalInputs


class ModelAttention(nn.Module):
    def __init__(self, input_size, output_size, key_size):
        super(ModelAttention, self).__init__()

        self.keyLayer = nn.Linear(input_size, key_size)
        self.queryLayer = nn.Linear(input_size, key_size)

        self.keyLayer2 = nn.Linear(output_size, key_size)
        self.queryLayer2 = nn.Linear(output_size, key_size)
        self.valueLayer2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        keys = self.keyLayer(x)
        querys = self.queryLayer(x)
        values = x

        raw_scores = torch.bmm(querys, torch.transpose(keys, 1, 2))
        scores = nn.functional.softmax(raw_scores, dim=2)
        outputs = torch.bmm(scores, values)

        keys2 = self.keyLayer2(outputs)
        querys2 = self.queryLayer2(outputs)
        values2 = self.valueLayer2(outputs)

        values2 = values2.div(torch.norm(values2, dim=2, keepdim=True))

        raw_scores2 = torch.bmm(querys2, torch.transpose(keys2, 1, 2))
        scores2 = nn.functional.softmax(raw_scores2, dim=2)
        finalOutputs = torch.bmm(scores2, values2)

        out = finalOutputs[0, -1, :]  # only need final output? or maybe use first for user preference...
        return out


class ModelAttentionNoLayers(nn.Module):
    def __init__(self, input_size, output_size, key_size):
        super(ModelAttentionNoLayers, self).__init__()

        self.keyLayer = nn.Linear(input_size, key_size)
        self.queryLayer = nn.Linear(input_size, key_size)

    def forward(self, x):
        keys = self.keyLayer(x)
        querys = self.queryLayer(x)
        values = x

        raw_scores = torch.bmm(querys, torch.transpose(keys, 1, 2))
        scores = nn.functional.softmax(raw_scores, dim=2)
        outputs = torch.bmm(scores, values)

        out = outputs[0, -1, :]  # only need final output? or maybe use first for user preference...
        return out


class ModelAttention2Layers(nn.Module):
    def __init__(self, input_size, output_size, key_size):
        super(ModelAttention2Layers, self).__init__()

        self.keyLayer = nn.Linear(input_size, key_size)
        self.queryLayer = nn.Linear(input_size, key_size)

        self.keyLayer2 = nn.Linear(output_size, key_size)
        self.queryLayer2 = nn.Linear(output_size, key_size)
        self.valueLayer2 = nn.Linear(output_size, output_size)

        self.keyLayer3 = nn.Linear(output_size, key_size)
        self.queryLayer3 = nn.Linear(output_size, key_size)
        self.valueLayer3 = nn.Linear(output_size, output_size)

    def forward(self, x):
        # layer 1
        keys = self.keyLayer(x)
        querys = self.queryLayer(x)
        values = x

        raw_scores = torch.bmm(querys, torch.transpose(keys, 1, 2))
        scores = nn.functional.softmax(raw_scores, dim=2)
        outputs = torch.bmm(scores, values)

        # layer 2
        keys2 = self.keyLayer2(outputs)
        querys2 = self.queryLayer2(outputs)
        values2 = self.valueLayer2(outputs)
        values2 = values2.div(torch.norm(values2, dim=2, keepdim=True))

        raw_scores2 = torch.bmm(querys2, torch.transpose(keys2, 1, 2))
        scores2 = nn.functional.softmax(raw_scores2, dim=2)
        hiddenLayer = torch.bmm(scores2, values2)

        # layer 3
        keys3 = self.keyLayer2(hiddenLayer)
        querys3 = self.queryLayer2(hiddenLayer)
        values3 = self.valueLayer2(hiddenLayer)
        values3 = values3.div(torch.norm(values3, dim=2, keepdim=True))

        raw_scores3 = torch.bmm(querys3, torch.transpose(keys3, 1, 2))
        scores3 = nn.functional.softmax(raw_scores3, dim=2)
        finalOutputs = torch.bmm(scores3, values3)

        out = finalOutputs[0, -1, :]
        return out


def eval_rank_and_loss(testing, model):
    tsize = len(testing)
    j = 0
    # lossvalues = []
    rankvalues = []
    HRat1 = []
    HRat3 = []
    print("evaluating average ranking and loss of target items according to model")

    for datum in testing:
        j += 1
        if (j % 5000 == 0):
            print(j, "out of", tsize)
        # get inputs, target, negative samples
        inputs, target = datum2inputs(datum)

        stackedInputs = stacker(inputs)
        output = model.forward(stackedInputs)

        output = output.cpu().detach().numpy()

        results = -1 * items @ output
        rankings = np.argsort(results)
        targetRank = np.where(rankings == datum['item'])[0] + 1

        HRat1.append(targetRank[0] == 1)
        HRat3.append(targetRank[0] <= 3)
        rankvalues.append(targetRank[0])

    df = pd.DataFrame(rankvalues, columns=["rank values"])
    df.to_csv("rankValuesTestDistribution.csv")

    tensorVals = torch.Tensor(rankvalues)
    median_rank = torch.median(tensorVals)
    test_rank = torch.mean(tensorVals)

    hr1 = torch.mean(torch.Tensor(HRat1))
    hr3 = torch.mean(torch.Tensor(HRat3))
    return test_rank, median_rank, hr1, hr3


def NLS_Loss(datum, output, target, negative_samples):
    negsamples = negative_random_samples(datum, total_samples=negative_samples)
    targetDot = torch.dot(target, output)
    out = torch.matmul(torch.cat(negsamples), output)
    ls = torch.logsumexp(out, 0) - targetDot
    return ls

def NCE_Loss(datum, output, target, negative_samples):
    negsamples = negative_contrastive_samples(datum, output, total_samples=negative_samples)
    targetDot = torch.dot(target, output)
    out = torch.matmul(torch.cat(negsamples), output)
    ls = torch.logsumexp(out, 0) - targetDot
    return ls

def NTE_Loss(datum, output, target, negative_samples):
    negsamples = negative_temperature_samples(datum, output, total_samples=negative_samples, temperature=1)
    targetDot = torch.dot(target, output)
    out = torch.matmul(torch.cat(negsamples), output)
    ls = torch.logsumexp(out, 0) - targetDot
    return ls

def NTE_Loss2(datum, output, target, negative_samples):
    negsamples = negative_temperature_samples(datum, output, total_samples=negative_samples, temperature=2)
    targetDot = torch.dot(target, output)
    out = torch.matmul(torch.cat(negsamples), output)
    ls = torch.logsumexp(out, 0) - targetDot
    return ls

def NTE_Loss3(datum, output, target, negative_samples):
    negsamples = negative_temperature_samples(datum, output, total_samples=negative_samples, temperature=3)
    targetDot = torch.dot(target, output)
    out = torch.matmul(torch.cat(negsamples), output)
    ls = torch.logsumexp(out, 0) - targetDot
    return ls

def NTE_Loss4(datum, output, target, negative_samples):
    negsamples = negative_temperature_samples(datum, output, total_samples=negative_samples, temperature=4)
    targetDot = torch.dot(target, output)
    out = torch.matmul(torch.cat(negsamples), output)
    ls = torch.logsumexp(out, 0) - targetDot
    return ls

def eval_rank_initial(testing):
    tsize = len(testing)
    j = 0
    lossvalues = []
    print("evaluating average initial rank of target item")
    for datum in testing:
        j += 1
        if (j % 5000 == 0):
            print(j, "out of", tsize)
        userVector = users[datum['user'], :]
        results = -1 * items @ userVector
        rankings = np.argsort(results)
        targetRank = np.where(rankings == datum['item'])[0] + 1
        lossvalues.append(targetRank)
    test_rank = torch.mean(torch.Tensor(lossvalues))
    return test_rank


# 6 epochs 50 additional neg samples - average ranking 1.73

torch.autograd.set_detect_anomaly(True)

training_ranks = []
testing_ranks = []
training_medians = []
testing_medians = []
training_losses = []
testing_losses = []
hrAt1sTrain = []
hrAt3sTrain = []
hrAt1sTest = []
hrAt3sTest = []



samples = [50]
models = [ModelAttention2Layers]
numCritiques = [5]
loss_funcs = [NTE_Loss4]
min_ranks = [20]

for critiqueLen in numCritiques:
    for min_rank in min_ranks:
        max_rank = min_rank+10
        numReplicas = 5
        dataset = generate_dataset(critiqueLen, numReplicas, min_rank, max_rank, randomize=True)

        training, testing = train_test_split(dataset, test_size=0.2)
        tsize = len(training)

        print("\n\n")
        testInitialAcc = eval_rank_initial(testing)
        print("average original testing position", testInitialAcc)
        print("\n")
        trainInitialAcc = eval_rank_initial(training)
        print("average original training position", trainInitialAcc)
        print("\n")
        for loss_func in loss_funcs:
            for sample in samples:
                for mdl in models:
                    numEpochs = 10
                    lr = 0.0001
                    model = mdl(input_size=vector_size, output_size=vector_size, key_size=25)
                    model = model.to(dev)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    print("\nStart training for sample", sample, "model", mdl, "loss function", loss_func, "and critique count ", critiqueLen)
                    for i in range(numEpochs):
                        print("\n")
                        print("Epoch ", i)
                        j = 0
                        random.shuffle(training)
                        for datum in training:
                            if (j % 5000 == 0):
                                print(j, "out of", tsize)
                            j += 1
                            # get inputs, target, negative samples

                            inputs, target = datum2inputs(datum)
                            datum["inputs"] = inputs
                            datum["target"] = target

                            stackedInputs = stacker(inputs)

                            output = model.forward(stackedInputs)
                            ls = loss_func(datum, output, target, sample)
                            optimizer.zero_grad()
                            ls.backward()
                            optimizer.step()

                        trainRank, trainMedian, hr1Train, hr3Train = eval_rank_and_loss(training, model)
                        testRank, testMedian, hr1Test, hr3Test = eval_rank_and_loss(testing, model)

                        print("training mean rank ", trainRank, "for sample ", sample)
                        print("testing mean rank ", testRank, "for sample ", sample)

                        print("training median rank ", trainMedian, "for sample ", sample)
                        print("testing median rank ", testMedian, "for sample ", sample)

                        print("training HR@1 ", hr1Train, "for sample ", sample)
                        print("testing HR@1 ", hr1Test, "for sample ", sample)

                        print("training HR@3 ", hr3Train, "for sample ", sample)
                        print("testing HR@3 ", hr3Test, "for sample ", sample)
                        print("End of epoch", i)

                        testing_ranks.append(testRank)
                        training_ranks.append(trainRank)

                        testing_medians.append(testMedian)
                        training_medians.append(trainMedian)

                        hrAt1sTrain.append(hr1Train)
                        hrAt3sTrain.append(hr3Train)
                        hrAt1sTest.append(hr1Test)
                        hrAt3sTest.append(hr3Test)


            print("End training")

dfResults = pd.DataFrame(list(zip(training_ranks, testing_ranks, training_medians, testing_medians, hrAt1sTrain, hrAt1sTest, hrAt3sTrain, hrAt3sTest)),
                         columns=['avg training rank', 'avg testing rank', 'training median', 'testing median', 'training hrat1', 'testing hrat1',
                                  'training hrat3',
                                  'testing hrat3'])

dfResults.to_csv("randomizedResults.csv")

'''
plt.plot(samples, training_ranks)
plt.plot(samples, testing_ranks)
plt.xlabel("Samples")
plt.ylabel("Rank")
plt.legend(["Training","Testing"])
plt.title("TARGET ITEM AVERAGE RANK")
plt.savefig("ranks_vs_samples")
plt.close()


plt.plot(samples, hrAt1sTrain)
plt.plot(samples, hrAt1sTest)
plt.xlabel("Epoch")
plt.ylabel("Final average Rank")
plt.legend(["Training","Testing"])
plt.title("TARGET ITEM HR@1")
plt.savefig("hrat1_vs_samples")
plt.close()
'''
