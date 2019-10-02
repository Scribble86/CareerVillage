import pandas as pd
import numpy as np
import json
import re
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
        #self.words = list(self.webster.keys())
        #self.wordIndex = self.makeIndex()
        #print(self.words)
        self.answerData = self._getTrainAnswers()
        self.answers = self.simplify(self.answerData,4)
        self.answeredQuestions = self._getAnsweredIDs()
        #self.featureset = list(self.webster.values())
        #print(self.featureset)
        self.tags = self._getTags()
        self.questionTags = self._getQuestionTags()
        self.questionData = self._getTrainQuestions()
        self.questionData = self.simplify(self.questionData, 3)
        self.questions = self.simplify(self.questionData, 4)
        self.answersDictionary = self.makeQAMatch(self.answers)

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

    #when similar questions are found, we now can find who answered those questions.
    def makeQAMatch(self, answers):
        getAnswerAuthor = {}
        length = len(answers.index)
        for row in range(0, length):
            questionID = answers.iat[row, 2]
            answerAuthor = answers.iat[row,1]
            if getAnswerAuthor.get(questionID) is None:
                getAnswerAuthor[questionID] = [answerAuthor]
            else:
                getAnswerAuthor[questionID].append(answerAuthor)
        return getAnswerAuthor

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
            qf.iat[every, 4] = self.addTags(qID, string)

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

    def questionFeatureDict(self, s):
        features = {}
        for word in s.split():
            if features.get(word) is None:
                features[word] = 1
            else:
                features[word] += 1
        return features

    #removes garbage values from string
    def simplifyString(self, inputString):
        outputString = ""
        try:
            translator = str.maketrans("\n#?.&,()^~{}[]+`'\":;-\\/_!","*********************    ")
            inputString = inputString.translate(translator)
            inputString = inputString.replace("*","")
            #remove html tags
            verboten = re.compile(r'<[^<]+?>')
            inputString = verboten.sub("", inputString)
            inputString = inputString.strip()
            for word in inputString.split():
                if self.webster.get(word) is not None:
                    outputString += " " + word
        except AttributeError:
            inputString = ""
        return outputString

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
    # def featureify(self, inputString):
    #     inputString = self.simplifyString(inputString)
    #     tokens = inputString.split()
    #     index = self.wordIndex
    #     #this next line takes a long time
    #     features = copy.deepcopy(self.featureset)
    #     for word in tokens:
    #         if index.get(word) is not None:
    #             features[index.get(word)] += 1
    #     return features

    def _getAnsweredIDs(self):
        length = len(self.answerData.index)
        answeredIDs = {}
        for row in range(0, length):
            if answeredIDs.get(self.answerData.iat[row,2]) is None:
                answeredIDs[self.answerData.iat[row,2]] = [self.answerData.iat[row,1]]
            else:
                answeredIDs[self.answerData.iat[row, 2]].append(self.answerData.iat[row,1])
        return answeredIDs

class processor:

    def __init__(self):
        self.source = unpack()

    def exec(self, low, high):
        #execute simlist through low-high range for some values picked from above high range
        questions =  self.source.questions
        length = len(questions.index)
        generate = randomGenerator(high+1, length)

        for example in range(0, 10):
            index = generate.__next__()
            sample = questions.iat[index, 3] + " " + questions.iat[example, 4]
            sampleID = questions.iat[index,0]
            similar = self.createSimList(sampleID, sample, low, high)
            recommendation = self.createRecommendation(similar)
            print(str(sampleID) + " :" + str(recommendation))
            self.analysis((sampleID, recommendation))


    #produce users likely to answer from respondants to most similar questions.
    def createRecommendation(self, simList):
        userList = []
        #this is not working. How to fix?
        if simList is not None:
            for each in simList:
                if each[2] is not None:
                    for user in each[2]:
                        userList.append(user)
        userList = list(set(userList))
        if userList.__sizeof__() > 5:
            userList = userList[0:5]
        return userList

    def createSimList(self, inputQuestionID, question, low, high):
        questions =  self.source.questions
        #length = len(questions.index)
        best = [[0],[0],[0],[0],[0]]
        for row in range(low, high):
            #following line should combine question
            thisQuestion = questions.iat[row,3] + " " + questions.iat[row,4]
            thisFeature = self.source.questionFeatureDict(thisQuestion);
            inputFeature = self.source.questionFeatureDict(question)
            sim = self.calculateSimilarity(inputFeature, thisFeature)
            questionID = questions.iat[row, 0]
            answerDict = self.source.answersDictionary.get(questionID)
            record = (sim, questionID, answerDict)
            for each in range(0, 5):
                if record[0] > best[each][0]:
                    temp = record
                    record = best[each]
                    best[each] = temp
        return best

    def calculateSimilarity(self, x, y):
        one = []
        two = []

        if len(x) > len(y):
            a = x
            b = y
        else:
            a = y
            b = x

        one = list(a.values())
        for each in a.keys():
            if b.get(each) is not None:
                two.append(float(a[each]))
            else:
                two.append(float(0))

        numerator = np.dot(one, two)
        denominator = np.sqrt(np.dot(one,one))*np.sqrt(np.dot(two,two))
        if denominator != 0:
            sim = round(numerator/float(denominator), 3)
        else:
            sim = 0.0;
        return sim


    def analysis(self, tuple):
        match = False
        if tuple[1] is not None:
            for each in tuple[1]:
                if self.source.answeredQuestions.get(each) is not None:
                    print(str(tuple[0]) + " " + str(each) + " " + "Match!")
                    match = True
        if not match:
            print(str(tuple[0]) + " " + "no match!")
        return match


output = processor()
for all in range(0, 25):
    output.exec(0, 20000)


#test = unpack()

#teststr = "<li> <p>test string issupposed to? include somme mistakes! #thug-life</p> </li>"
#teststr = test.simplifyString(teststr)
#result = test.countMistakes(teststr)
#result = test.featureify(teststr)

#for each in result.keys():
#    if result.get(each) > 0:
#        print(each + ": " + str(result.get(each)))