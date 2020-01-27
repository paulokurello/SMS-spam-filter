import numpy
import pandas
import string
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def readDataSet(filename, dropedColumns):
    dataSet = pandas.read_csv(filename,  encoding='latin-1')
    dataSet.drop(columns=dropedColumns, inplace=True)
    return dataSet


def preprocessMessage(message, lowerCase = False, stemWords=False, rmStopWords=False):
    if lowerCase:
        message = message.lower()
    message = message.replace('(', ' ')
    message = message.replace(')', ' ')
    words = message.split()
    punctFilter = str.maketrans('', '', string.punctuation)
    words = [w.translate(punctFilter) for w in words]
    if rmStopWords:
        stopWord = stopwords.words('english')
        for word in words:
            if word in stopWord:
                words.remove(word)
    if stemWords:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


class Classifier:
    def __init__(self, trainingSet, alpha=1):
        self.messages = trainingSet['message']
        self.labels = trainingSet['value']
        self.messageCount = self.messages.size
        self.spamCount = self.labels.value_counts()[1]
        self.hamCount = self.messageCount - self.spamCount
        self.alpha = alpha
        self.wordsCount = 0
        self.wordsSpamCount = dict()
        self.wordsHamCount = dict()
        self.spamWordsCount = 0
        self.hamWordsCount = 0
        self.wordsInSet = dict()
        self.probWordsSpam = dict()    #p(w|spam)
        self.probWordsHam = dict()      #p(w|ham)
        self.probSpam = self.spamCount / self.messageCount   #p(spam)
        self.probHam = 1 - self.probSpam    #p(ham)
        self.probWords = dict()             #p(w)
        self.idf = dict()
        self.tfidfSumAll = 0
        self.tfidfSumHam = 0
        self.tfidfSumSpam = 0

    def calcParametersBOW(self, lowerCase = False, stemWords=False, rmStopWords=False):
        for i in range(self.messageCount):
            words = preprocessMessage(self.messages.iloc[i], lowerCase, stemWords, rmStopWords)
            for word in words:
                if self.labels.iloc[i]:
                    self.wordsSpamCount[word] = self.wordsSpamCount.get(word, 0) + 1
                    self.spamWordsCount += 1
                else:
                    self.wordsHamCount[word] = self.wordsHamCount.get(word, 0) + 1
                    self.hamWordsCount += 1
                self.wordsInSet[word] = self.wordsInSet.get(word, 0) + 1
                self.wordsCount += 1
        for word in self.wordsInSet:
            self.probWords[word] = (self.wordsInSet[word] + self.alpha) / (self.wordsCount + \
                                    len(list(self.wordsInSet.keys()))*self.alpha)
        for word in self.wordsSpamCount:
            self.probWordsSpam[word] = (self.wordsSpamCount[word] + self.alpha) / (self.spamWordsCount + \
                                        len(list(self.wordsSpamCount.keys()))*self.alpha)
        for word in self.wordsHamCount:
            self.probWordsHam[word] = (self.wordsHamCount[word] + self.alpha) / (self.hamWordsCount + \
                                        len(list(self.wordsHamCount.keys()))*self.alpha)

    def calcParametersTFIDF(self, lowerCase = False, stemWords=False, rmStopWords=False):
        for i in range(self.messageCount):
            label = self.labels.iloc[i]
            words = preprocessMessage(self.messages.iloc[i], lowerCase, stemWords, rmStopWords)
            for word in words:
                filter(lambda a: a != word, words)
                if label == 1:
                    self.wordsSpamCount[word] = self.wordsSpamCount.get(word, 0) + 1
                    self.spamWordsCount += 1
                else:
                    self.wordsHamCount[word] = self.wordsHamCount.get(word, 0) + 1
                    self.hamWordsCount += 1
                self.wordsInSet[word] = self.wordsInSet.get(word, 0) + 1
                self.wordsCount += 1
        for word in self.wordsInSet:
            self.idf[word] = math.log10(self.messageCount / (self.wordsSpamCount.get(word, 0) + \
                                                             self.wordsHamCount.get(word, 0)))
        for word in self.wordsInSet:
            self.tfidfSumAll += (self.wordsHamCount.get(word, 0) + self.wordsSpamCount.get(word, 0)) * self.idf[word]
            self.tfidfSumHam += self.wordsHamCount.get(word, 0) * self.idf[word]
            self.tfidfSumSpam += self.wordsSpamCount.get(word, 0) * self.idf[word]
        for word in self.wordsInSet:
            self.probWords[word] = (self.wordsSpamCount.get(word, 0) + self.wordsHamCount.get(word, 0)) * \
                                   self.idf[word] / self.tfidfSumAll
            self.probWordsHam[word] = (self.wordsHamCount.get(word, 0) * self.idf[word] + self.alpha) / \
                                      (self.tfidfSumHam + self.alpha * self.hamWordsCount)
            self.probWordsSpam[word] = (self.wordsSpamCount.get(word, 0) * self.idf[word] + self.alpha) / \
                                       (self.tfidfSumSpam + self.alpha * self.spamWordsCount)

    def TestMessages(self, testingSet, lowerCase = False, stemWords=False, rmStopWords=False):
        self.tmpMsg = testingSet['message']
        self.tmpLab = testingSet['value']
        self.tmpcount = self.tmpMsg.size
        self.correct = 0
        predictionList = list()
        for i in range(self.tmpcount):
            label = self.tmpLab.iloc[i]
            words = preprocessMessage(self.tmpMsg.iloc[i],  lowerCase, stemWords, rmStopWords)
            SpamProb = self.probSpam
            HamProb = self.probHam
            for word in words:
                filter(lambda a: a != word, words)
                if self.probWordsSpam.get(word, 0) == 0:
                    SpamProb = SpamProb * self.alpha / (self.alpha * self.spamWordsCount)
                else:
                    SpamProb *= self.probWordsSpam[word]
                if self.probWordsHam.get(word, 0) == 0:
                    HamProb = HamProb * self.alpha / (self.alpha * self.hamWordsCount)
                else:
                    HamProb *= self.probWordsHam[word]
            if HamProb > SpamProb:
                prediction = 0
            else:
                prediction = 1
            predictionList.append(prediction)
        return predictionList


def statistics(labels, predictions):
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
    for i in range(len(labels)):
        truePos += int(labels.iloc[i] == 1 and predictions[i] == 1)
        trueNeg += int(labels.iloc[i] == 0 and predictions[i] == 0)
        falsePos += int(labels.iloc[i] == 0 and predictions[i] == 1)
        falseNeg += int(labels.iloc[i] == 1 and predictions[i] == 0)
    precision = truePos / (truePos + falsePos)
    recall = truePos / (truePos + falseNeg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
    result = "Precision: ," + str(precision) + ", Recall: ," + str(recall) + ", F-score: ," + str(Fscore) + ", Accuracy: ," + str(accuracy) + ', \n'
    plik.write(result)


data = readDataSet('spam.csv', ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data.rename({'v1': 'value', 'v2': 'message'}, axis=1, inplace=True)
data['value'] = data['value'].map({'ham': 0, 'spam': 1})
plik = open('TFIDF---rmStop', 'w')


for k in range(50):
    trainIndex, testIndex = list(), list()
    for i in range(data.shape[0]):
        if numpy.random.uniform(0, 1) < 0.75:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = data.loc[trainIndex]
    testData = data.loc[testIndex]

    test = Classifier(trainData)
    test.calcParametersTFIDF(False, False, True)
    results = list()
    results = test.TestMessages(testData)
    statistics(testData['value'], results)

plik.close()
