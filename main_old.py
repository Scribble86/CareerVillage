import pandas as pd
import numpy as np
import sklearn as sk
import json
import re
import copy
import random

#this is an iterator that can simulate random selection without replacement.
class randomGenerator:

    def __init__(self, low, high):
        self.lo = low
        self.hi = high
        self.index = 0
        self.values = random.sample(list(range(low, high)),(high-low))

    def __next__(self):
        if self.index < self.hi:
            rval = self.values[self.index]
            self.index += 1
            return rval;
        else:
            raise StopIteration

    def __iter__(self):
        return self

class unpack:

    #load files we're using
    def __init__(self):
        #dictionary of all words
        self.webster = self._getWords()
        #list of all words
        self.words = list(self.webster.keys())
        self.wordIndex = self.makeIndex()
        #print(self.words)
        self.answerData = self._getTrainAnswers()
        self.answers = self.simplify(self.answerData,4)
        self.answeredQuestions = self.getAnsweredIDs()
        self.featureset = list(self.webster.values())
        #print(self.featureset)
        self.tags = self._getTags()
        self.questionTags = self._getQuestionTags()
        self.questionData = self._getTrainQuestions()
        self.questions = self.simplify(self.questionData, 4)

    #load dictionary and remove stop words
    def _getWords(self):
        with open("words_dictionary.json") as f:
            data = json.load(f)
        with open("stop.json") as s:
            stop = json.load(s)
        for key in stop.keys():
            if key in data:
                del data[key]
        return data

    def makeIndex(self):
        index = {}
        val = 0
        for word in self.words:
            index[word] = val
            val += 1
        return index

    #remove garbage values from strings in loaded files (as dataframes)
    def simplify(self, file, column):
        length = len(file.index)
        generator = randomGenerator(0, length)
        for every in range(0,length):
            index = generator.__next__()
            result = file.iat[index, column]
            #print(str(index) + " " + str(result))
            file.iat[index, column] = self.simplifyString(result)
        return file

    #function that adds tags to all questions in set to add to feature vector.
    #unneeded as the words already exist in the string.
    def addAllTags(self, qf):
        length = len(qf.index)
        for every in range(0, length):
            string = qf.iat[every, 4]
            qID = qf.iat[every, 0]
            qf.iat[every, 4] = self.addTags(qID, string);

    #function to add tags from tag collection to a specific question. it turns out that tags already exist in the questin
    #so this is mostly unnecessary at this stage.
    def addTags(self, questionID, string):
        tags = self.questionTags.get(questionID)
        definitions = self.tags
        if tags is not None:
            for tag in tags:
                result = definitions.get(tag)
                if result is not None:
                    add = str(result)
                    string = string + " " + add
        return string

    #functions below load files into dataframes
    def _getTrainQuestions(self):
        with open("careervillage/questions.csv") as f:
            file = pd.read_csv(f)
            return file

    def _getTrainAnswers(self):
        with open("careervillage/answers.csv") as f:
            file = pd.read_csv(f)
        return file

    def _getTags(self):
        with open("careervillage/tags.csv") as f:
            file = pd.read_csv(f)
            length = len(file.index)
            tagsdict = {}
            for row in range(0, length):
                tag = file.iat[row,1]
                key = file.iat[row,0]
                if tagsdict.get(key) is not None:
                    tagsdict[key] = [tagsdict[key]].append(tag)
                else:
                    tagsdict[key] = tag
            return tagsdict

    def _getQuestionTags(self):
        with open("careervillage/tag_questions.csv") as f:
            file = pd.read_csv(f)
            length = len(file.index)
            questiontagsdict = {}
            for row in range(0, length):
                key = file.iat[row,1]
                tag = file.iat[row,0]
                if questiontagsdict.get(key) is not None:
                    questiontagsdict[key].append(tag)
                else:
                    questiontagsdict[key] = [tag]
            #print(questiontagsdict)
            return questiontagsdict

    #removes garbage values from string
    def simplifyString(self, inputString):
        try:
            translator = str.maketrans("\n#?.&,()^~{}[]+`'\":;-\\/_!","*********************    ")
            inputString = inputString.translate(translator)
            inputString = inputString.replace("*","")
            #remove html tags
            verboten = re.compile(r'<[^<]+?>')
            inputString = verboten.sub("", inputString)
            inputString = inputString.strip()
        except AttributeError:
            inputString = ""
        return inputString

    #currently unused function to count misspelled words in a string
    #note that currently it will count stop words (common words) as mistakes
    def countMistakes(self,inputString):
        input = inputString.lower()
        input = input.split()
        output = ""
        errors = 0
        for word in input:
            if self.webster.get(word) is None:
                errors += 1
            else:
                output += " " + word
        output = output.strip()
        return (output, errors)

    #transforms input string to dictionary with words as keys and values as frequency
    def featureify(self, inputString):
        inputString = self.simplifyString(inputString)
        tokens = inputString.split()
        index = self.wordIndex
        #this next line takes a long time
        features = copy.deepcopy(self.featureset)
        for word in tokens:
            if index.get(word) is not None:
                features[index.get(word)] += 1
        return features

    def getAnsweredIDs(self):
        length = len(self.answerData.index)
        answeredIDs = []
        for row in range(0, length):
            answeredIDs.append(self.answerData.iat[row,0])
        return answeredIDs

class processor:

    def __init__(self, low, high):
        #convert data from unpack to create a set of feature vectors
        self.source = unpack()
        self.question_ID_List = self.source.answeredQuestions
        self.question_set = []

    def createQuestionSet(self, low, high):
        generator = randomGenerator(low, high)
        questionset = []
        for each in range(low,high):
            feature = self.source.featureify(self.source.questions.iat[each,4])
            questionset.append(feature)
            print("Working on featureset " + str(each) + " of " + str(high))
        return questionset


output = processor(0, 2000)
questions = output.createQuestionSet(0, len(output.source.questions.index))

with open("features.dat", "w") as out:
    json.dump(questions, out)


outputfile = "features.dat"



#test = unpack()

#teststr = "<li> <p>test string issupposed to? include somme mistakes! #thug-life</p> </li>"
#teststr = test.simplifyString(teststr)
#result = test.countMistakes(teststr)
#result = test.featureify(teststr)

#for each in result.keys():
#    if result.get(each) > 0:
#        print(each + ": " + str(result.get(each)))