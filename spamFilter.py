import pandas
import string
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


class BagOfWordsClassifier:
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

    def calcParameters(self):
        for i in range(self.messageCount):
            label = self.labels[i]
            words = preprocessMessage(self.messages[i])
            for word in words:
                if label == 1:
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


data = readDataSet('spam.csv', ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data.rename({'v1': 'value', 'v2': 'message'}, axis=1, inplace=True)
print(data.head())
print(data['value'].value_counts())
data['value'] = data['value'].map({'ham': 0, 'spam': 1})
print(data.head())

test = BagOfWordsClassifier(data)
test.calcParameters()
print(data['value'][250])
print(data['message'][250])






