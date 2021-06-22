import csv, numpy

def openFile(name):
    data = []
    with open(name) as file:
        reader = csv.reader(file)
        for line in reader:
            data.append([])
            data[-1] = list(numpy.fromstring(line[0], dtype=int, sep=' '))
        return data

def createClassification(instances):
    spam_count = 0
    not_spam_count = 0
    spam = []
    not_spam = []

    for instance in instances:
        if (instance[-1] == 1):
            spam_count += 1
            spam.append(instance)
        elif (instance[-1] == 0):
            not_spam_count += 1
            not_spam.append(instance)
        else:
            raise TypeError('Class can either be 0 or 1, got: ' + instance[-1])

    feature_count = len(instances[0][0:-1]) # Remove class column
    spam_probs = createTable(spam, feature_count)
    not_spam_probs = createTable(not_spam, feature_count)

    for i in range(feature_count):
        print('Feature #' + str(i))
        print('P(f|Spam)', spam_probs[i], '\t\tP(not f|Spam)', 1 - spam_probs[i])
        print('P(f|NotSpam)', not_spam_probs[i], '\tP(not f|Spam)', 1 - not_spam_probs[i])
        print('---------------------------------')

    return spam_probs, spam_count, not_spam_probs, not_spam_count, len(instances)

def calcScore(instance, probs, count, total):
    score = 1
    for i in range(len(instance)):
        if (instance[i] == 1):
            score *= probs[i]
        elif (instance[i] == 0):
            score *= 1 - probs[i]
        else:
            raise TypeError('Class can either be 0 or 1, got: ' + instance[-1])

    score *= count / total
    return score

def createTable(instances, feature_count):
    instance_count = len(instances)
    feature_counts = [0] * feature_count
    for row in instances:
        for i in range(len(feature_counts)):
            if (row[i] == 1):
                feature_counts[i] += 1

    prob_feature_given_class = [0] * len(feature_counts)
    for i, feature in enumerate(feature_counts):
        if feature == 0:
            prob_feature_given_class[i] = (feature + 1) / (instance_count + 1)
        else:
            prob_feature_given_class[i] = feature / instance_count

    return prob_feature_given_class

def predictClass(instances, spam_probs, spam_count, not_spam_probs, not_spam_count, total):
    for i in range(len(instances)):
        spam_score = calcScore(instances[i], spam_probs, spam_count, total)
        not_spam_score = calcScore(instances[i], not_spam_probs, not_spam_count, total)

        result = 1 if spam_score > not_spam_score else 0 

        print('Instance #', i)
        print('Email is:\t', ('Spam' if result == 1 else 'Not spam'))
        print('Result:\t\t', result)
        print('Spam score:\t', spam_score)
        print('Not Spam score:\t', not_spam_score)
        print('--------------------')

def main():
    data = openFile('spamLabelled.dat')
    spam_probs, spam_count, not_spam_probs, not_spam_count, total = createClassification(data)
    instances = openFile('spamUnlabelled.dat')
    predictClass(instances, spam_probs, spam_count, not_spam_probs, not_spam_count, total)

if __name__ == "__main__":
    main()
