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


data = readDataSet('spam.csv', ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data.rename({'v1': 'value', 'v2': 'message'}, axis=1, inplace=True)
print(data.head())
print(data['value'].value_counts())
data['value'] = data['value'].map({'ham': 0, 'spam': 1})
print(data.head())

rowCount = data['value'].count()

for i in preprocessMessage(data['message'][2]):
    print(i)





