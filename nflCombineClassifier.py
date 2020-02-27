 #!/usr/bin/env python
""" NFL Combine Draft Classifier

 This code uses NFL combine data to predict if an incoming player will
 will make it into the NFL via the NFL draft, based exclusively on his combine stati.
 This current code requires you to first preProcess combine.txt in main if you have not already done so.
 After pre-processing is done, you may comment out that section, unless changes are made to pre-processing algorithm.
 
 The classifier used is Draft round (0-7, 0 being not drafted).

Example:
    Before running this file, it is assumed that you will also have 'combine.txt' in the same directory path.
	It is downloaded from here in csv form: http://nflsavant.com/dump/combine.csv?year=2015
	This site currently only has data from 1999 to 2015.
	
        $ python nflCombineClassifier.py
"""

import matplotlib.pyplot as pyplot
import numpy
import csv
import copy
import random
import math
import operator
import tabulate

def read_csv(filename):
    """Reads in a csv file and returns a table as a list of lists (rows)"""
    the_file = open(filename)
    the_reader = csv.reader(the_file, dialect='excel')
    table = []
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    return table

def write_csv(fileName, table):
    the_file = open(fileName, 'w+')
    the_writer = csv.writer(the_file, dialect='excel')
    for row in table:
        the_writer.writerow(row)

def get_column(table, index):
    """Returns all non-null values in table for given index"""
    vals = []
    for row in table:
        if row[index] != "NA":
            vals.append(float(row[index]))
    return vals

def get_column_with_NA(table, index):
    """Returns all values in table for given index"""
    vals = []
    for row in table:
        vals.append(float(row[index]))
    return vals

def get_column_string(table, index):
    vals = []
    for row in table:
        if row[index] != "NA":
            vals.append(row[index])
    return vals

def group_by(table, att_index):
    # create unique list of grouping values
    grouping_values = []
    for row in table:
        value = row[att_index]
        if value not in grouping_values:
            grouping_values.append(value)
    grouping_values.sort()
    
    # create list of n empty partitions
    result = [[] for _ in range(len(grouping_values))]
    
    # add rows to each partition
    for row in table:
        result[grouping_values.index(row[att_index])].append(row[:])
    return result

def frequencies(xs):
 """Returns a unique, sorted list of values in xs and occurrence counts for each value"""
 ys = sorted(xs)
 values, counts = [], []
 for y in ys:
    if y not in values:
       values.append(y)
       counts.append(1)
    else:
        counts[-1] += 1
 return values, counts

def createPieChart(table, column, indexName):
    col = get_column_with_NA(table, column)

    values, count = frequencies(col)
    pyplot.figure(figsize=(8,8))
    pyplot.title(indexName)
    pyplot.pie(count, labels=values, autopct='%1.1f%%') #format pcts as 12.3%, etc
    pyplot.savefig(indexName + 'PieChart.png')
    pyplot.close()

def boxPlot(table, xIndex, yindex, xName, yName, ystart, ylimit):
    lists = []
    xLabels = []
    tables = group_by(table, xIndex)
    for SubTable in tables:
        # add each different SubTable xIndex to xLabels
        xLabels.append(SubTable[0][xIndex])
        # create a list of MPG data for each subTable
        lists.append(get_column(SubTable, yindex))
        
    pyplot.figure()
    pyplot.xlabel(xName)
    pyplot.ylabel(yName)
    pyplot.title(yName + " by " + xName)
    pyplot.boxplot(lists)
    pyplot.yticks(numpy.arange(18, 45, 1))
    pyplot.ylim(ystart, ylimit)
    pyplot.xticks(numpy.arange(1,len(xLabels)+1), xLabels)

    pyplot.savefig(xName + yName + 'BoxPlot.png')
    pyplot.close()

# removes attributes at each index in indexes
def remove_unwanted_attr(table, indexes):
    newTable = copy.deepcopy(table)
    for i in range(len(newTable)):
        for indices in indexes:
            del newTable[i][indices]
    return newTable

def minimum(table, index):
    return min(get_column(table, index))

def maximum(table, index):
    return max(get_column(table, index))

def mid(table, index):
    return (minimum(table,index)+maximum(table,index))/2.0

def avg(table, index):
    column = get_column(table, index)
    if len(column) == 0:
        return 0
    else:
        return (sum(column)/float(len(column)))

def median(table, index):
    column = get_column(table, index)
    column.sort()
    mid = len(column)/2
    if len(column) % 2 == 0:
        return (column[mid - 1] + column[mid])/2
    else:
        return column[mid]

def filteredAvg(table, index, rowIndex):
    """filters the table on position played, then averages the filtered table"""
    filteredTable = []
    for row in table:
        if row[3] == table[rowIndex][3] and row[index] > 0:
            filteredTable.append(row)
    if not filteredTable:
        return avg(table, index)

    average = round(avg(filteredTable, index), 3)
    if average == 0.0:
       print row[0], "", row[3]
    return average

def replaceNAwithFilteredAVG(table, indexes):
    for row in range(len(table)):
        for column in indexes:
            if float(table[row][column]) <= 0.0:
                table[row][column] = filteredAvg(table, column, row)

def removeNA(table, indexes):
    for i in range(len(table)-1, -1, -1):
        for column in indexes:
            if float(table[i][column]) <= 0.0:
                del table[i]

def print_statistics(table, indexes, attr_names):
    header = ["attribute", "min", "max", "mid", "avg", "med"]
    printTable = []
    for i in range(len(indexes)):
        printTable.append([attr_names[i], minimum(table, indexes[i]), maximum(table, indexes[i]), mid(table, indexes[i]), avg(table, indexes[i]), median(table, indexes[i])])
    print tabulate.tabulate(printTable, header, tablefmt="rst")

#-------KNN SECTION-------

def get_column_by_position(table, column, position):
    col = []
    for i in range(len(table)):
        if table[i][3] == position and table[i][column] != 0.0:
            col.append(float((table[i][column])))
    return col

def grab_positions(table):
    positions = []
    for row in table:
        if row[3] not in positions:
            positions.append(row[3])
    return positions

def normalize(Xs):
    x_max = max(Xs)
    x_min = min(Xs)
    x_maxmin = float(x_max) - float(x_min)
    normalized = []

    for i in range(len(Xs)):
        x = Xs[i]
        if x_maxmin == 0.0:
            normalized.append(1)
        else:
            normalized.append(round(((x - x_min) / x_maxmin), 3))
    return normalized

def select_class_label(row_distances):
    nearest = []
    yes, no = 0, 0
    for row in row_distances:
        nearest.append(row[1])
    yes, no = 0, 0
    for label in nearest:
        if label[-1] > 0.0:
            yes += 1
        else:
            no += 1
    if yes >= no:
        return "Yes"
    else:
        return "No"

    #values, counts = frequencies(get_column_with_NA(nearest, 7))
    #return values[counts.index(max(counts))]


def k_nn_classifier(training_set, numInstances, instance, k):
    row_distances = []
    for row in training_set:
        row_distances.append([distance(row, instance, numInstances), row])
    row_distances.sort(key = operator.itemgetter(0))
    label = select_class_label(row_distances[0:k])
    return label

""" 
   MUST be sorted such that all attributes are from same instance,
   The instances will be retrieved in the same order.
"""
def normalized_for_knn(table):
    sortedTable = copy.deepcopy(table)
    sortedTable.sort(key = operator.itemgetter(3))
    normalizedTable = []
    positions = grab_positions(sortedTable)

    normalized_40yd = normalize_by_position(sortedTable, positions, 5)
    for i in normalized_40yd:
        normalizedTable.append([i])

    normalized_weight = normalize_by_position(sortedTable, positions, 4)
    for i in range(len(normalized_weight)):
        normalizedTable[i].append(normalized_weight[i])

    normalized_height = normalize_by_position(sortedTable, positions, -1)
    for i in range(len(normalized_height)):
        normalizedTable[i].append(normalized_height[i])

    normalized_vert = normalize_by_position(sortedTable, positions, 8)
    for i in range(len(normalized_vert)):
        normalizedTable[i].append(normalized_vert[i])

    normalized_broad = normalize_by_position(sortedTable, positions, 9)
    for i in range(len(normalized_broad)):
        normalizedTable[i].append(normalized_broad[i])

    normalized_bench = normalize_by_position(sortedTable, positions, 10)
    for i in range(len(normalized_bench)):
        normalizedTable[i].append(normalized_bench[i])


    pick_column = get_column(sortedTable, 11)
    for i in range(len(normalizedTable)):
        normalizedTable[i].append(pick_column[i])

    return normalizedTable

def normalized_for_knn_2(table):
    sortedTable = copy.deepcopy(table)
    sortedTable.sort(key = operator.itemgetter(3))
    normalizedTable = []
    positions = grab_positions(sortedTable)

    normalized_40yd = normalize(get_column(sortedTable, 5))
    for i in normalized_40yd:
        normalizedTable.append([i])

    normalized_weight = normalize(get_column(sortedTable, 4))
    for i in range(len(normalized_weight)):
        normalizedTable[i].append(normalized_weight[i])

    normalized_height = normalize(get_column(sortedTable, -1))
    for i in range(len(normalized_height)):
        normalizedTable[i].append(normalized_height[i])

    normalized_vert = normalize(get_column(sortedTable, 8))
    for i in range(len(normalized_vert)):
        normalizedTable[i].append(normalized_vert[i])

    normalized_broad = normalize(get_column(sortedTable, 9))
    for i in range(len(normalized_broad)):
        normalizedTable[i].append(normalized_broad[i])

    normalized_bench = normalize(get_column(sortedTable, 10))
    for i in range(len(normalized_bench)):
        normalizedTable[i].append(normalized_bench[i])

    pick_column = get_column(sortedTable, 11)
    for i in range(len(normalizedTable)):
        normalizedTable[i].append(pick_column[i])
    return normalizedTable

def normalize_by_position(table, position_played, attribute):
    normalized_attribute = []
    for i in range(len(position_played)):
        col = get_column_by_position(table, attribute, position_played[i])
        normalized_attribute += normalize(col)
    return normalized_attribute

def distance(row, instance, numInstances):
    distance = 0.0
    for i in range(numInstances-1):
        distance += (float(row[i])-float(instance[i]))**2
    return math.sqrt(distance)

def KNN(table):
    normalizedTable = normalized_for_knn_2(table)
    for i in range(5):
        index = random.randint(0, len(normalizedTable) - 1)
        print "instance: ",table[index]
        predictClass = k_nn_classifier(normalizedTable, 8, normalizedTable[index], 10)
        actual = table[index][11]
        print "class: ",predictClass,", actual: ",actual
    print ""

#------ end KNN ------

def create_stratified_kfolds(table):
    disc_table = copy.deepcopy(table)
    # discretized instances in disc table 
    disc_table.sort(key = operator.itemgetter(-1))
    folds = [[] for _ in range(10)]
    index = 0
    for row in disc_table:
        folds[index].append(row)
        index = (index + 1) % 10
    return folds

def binClassifier(prediction):
    if prediction == 0.0:
        return 0
    elif prediction == 1.0 or prediction == 2.0:
        return 1
    elif prediction == 3.0 or prediction == 4.0 or prediction == 5.0:
        return 2
    elif prediction == 6.0 or prediction == 7.0:
        return 3
    else:
        return -1

def isRight(prediction, actual):
    if binClassifier(prediction) == binClassifier(actual):
        return 1
    else:
        return 0

def holdout_partition(table):
    # randomize the table
    randomized = table[:] # copy the table
    n = len(table)
    for i in range(n):
        # pick an index to swap
        j = random.randint(0, n-1) # random int [0,n-1] inclusive
        randomized[i], randomized[j] = randomized[j], randomized[i]
    # return train and test sets
    n0 = (n * 2)/3
    return randomized[0:n0], randomized[n0:]

def random_subsampling_validation_knn(table):
    accuracy = 0.0
    true_positives_total = 0.0
    total_predictions = 0
    newTable = normalized_for_knn(table)
    for i in range(10):
        training_set, test_set = holdout_partition(newTable)

        for index in range(len(test_set)):
            predictClass = k_nn_classifier(training_set, 7, test_set[index], 20)
            actual = test_set[index][-1]
            total_predictions += 1
            if (predictClass == "Yes" and actual > 0.0):
                true_positives_total += 1

    accuracy = round((true_positives_total/total_predictions), 3)
    standardError = round((math.sqrt(accuracy*(1 - accuracy)/len(test_set))), 3)
    return accuracy, standardError

def stratified_kfold_validation_knn(table):
    accuracy = 0.0
    true_positives_total = 0.0
    total_predictions = 0
    newTable = normalized_for_knn(table)
    folds = create_stratified_kfolds(newTable)
    for i in range(10):
        test_set = folds[i]
        training_set = []
        for j in range(10):
            if (i != j):
                training_set += folds[j]

        #normalized_training_set = normalized_for_knn(training_set)
        #normalized_test_set = normalized_for_knn(test_set)

        for index in range(len(test_set)):
            predictClass = k_nn_classifier(training_set, 7, test_set[index], 20)
            actual = test_set[index][-1]
            total_predictions += 1
            if (predictClass == "Yes" and actual > 0.0):
                true_positives_total += 1

    accuracy = round((true_positives_total/total_predictions), 3)
    standardError = round((math.sqrt(accuracy*(1 - accuracy)/len(test_set))), 3)
    return accuracy, standardError

#
# ------- Decision Tree Classifier -------
#

def printTreeLogic(tree, rule):
    if (isinstance(tree[0], list)):
        maxStat = tree[0][1]
        maxClass = tree[0][0]
        for stat in tree:
            if stat[1] > maxStat:
                maxClass = stat[0]
                maxStat = stat[1]
        rule += "THEN class = "+ str(maxClass)
        print rule
        return rule
    rule += "IF Attribute " + str(tree[0]['Attribute'])
    for i in range(1, len(tree)):
        printTreeLogic(tree[i][1], rule + " == Value " + str(tree[i][0]['Value']) + " And ")

def eNew(instances, index, class_index):
    total = float(len(instances))
    eNew = 0
    partition_table = group_by(instances, index)
    for table in partition_table:
        class_labels = [0 for _ in range(8)]
        for row in table:
            class_labels[int(row[class_index]) - 1] += 1

        sum_eNew = 0
        for label in class_labels:
            if (label != 0):
                sum_eNew -= label*math.log(label, 2)
        eNew += -1*(sum_eNew)*(len(table)/total)
    return eNew

def select_attribute(instances, att_indexes, class_index):
    E = eNew(instances, att_indexes.keys()[0], class_index)
    att_index = att_indexes.keys()[0]
    for i in range(1, len(att_indexes)-1):
        temp_eNew = eNew(instances, att_indexes.keys()[i], class_index)
        if(temp_eNew < E):
            E = temp_eNew
            att_index = att_indexes.keys()[i]
    return att_index

def in_same_class(instances, class_index):
    label = instances[0][class_index]
    for row in instances:
        if row[class_index] != label:
            return False
    return True

def partition_stats(instances, class_index):
    partition_stats_list = []
    total = float(len(instances))
    tables = group_by(instances, class_index)
    for table in tables:
        partition_stats_list.append([table[0][class_index], len(table), total])
    return partition_stats_list

def partition_instances(instances, att_index):
    tables = group_by(instances, att_index)
    partition_instances_list = {}
    for table in tables:
        partition_instances_list[table [0][att_index]] = table # {table[0][att_index]:table}
    return partition_instances_list

def tdidt(instances, att_indexes, class_index):
    tree = []
    if (in_same_class(instances, class_index)):
        return partition_stats(instances, class_index)
    if (len(att_indexes) == 0):
        return partition_stats(instances, class_index)
    att_index = select_attribute(instances, att_indexes, class_index)
    partition_list = partition_instances(instances, att_index)
    if (len(partition_list) != att_indexes[att_index]):
        return partition_stats(instances, class_index)
    tree.append({'Attribute':att_index})
    att_indexes2 = copy.deepcopy(att_indexes)
    del att_indexes2[att_index]
    for i, v in partition_list.iteritems():
        tree.append([{'Value':i},tdidt(v, att_indexes2, class_index)])
    return tree

def decision_tree_classification(tree, row):
    if(isinstance(tree[0], dict)):
        for i in range(1, len(tree)):
            if (row[tree[0]['Attribute']] == tree[i][0]['Value']):
                return decision_tree_classification(tree[i][1], row)

    else:
        maxStat = tree[0][1]
        maxClass = tree[0][0]
        for stat in tree:
            if stat[1] > maxStat:
                maxClass = stat[0]
                maxStat = stat[1]
        return maxClass

def discretize_forty(table):
    for row in table:
        if row[5] <= 4.4:
            row[5] = 0.0
        elif row[5] <= 4.6:
            row[5] = 1.0
        elif row[5] <= 4.8:
            row[5] = 2.0
        elif row[5] <= 5.0:
            row[5] = 3.0
        else:
            row[5] = 4.0

def discretize_on_pivot(table, pivot, pivot_index):
    newTable = copy.deepcopy(table)
    for row in newTable:
        if row[pivot_index] < pivot:
            row[pivot_index] = 0
        else:
            row[pivot_index] = 1
    return newTable

def discretize_cont(table, pivot_index):
    E = eNew(discretize_on_pivot(table, table[0][pivot_index], pivot_index), pivot_index, 11)
    pivot = 0
    for i in range(1, len(table)):
        instances = discretize_on_pivot(table, table[i][pivot_index], pivot_index)
        temp_eNew = eNew(instances, pivot_index, 11)
        if(temp_eNew < E):
            E = temp_eNew
            pivot = table[i][pivot_index]

    return discretize_on_pivot(table, pivot, pivot_index)

def decision_tree_classification(tree, row):
    if(isinstance(tree[0], dict)):
        for i in range(1, len(tree)):
            if (row[tree[0]['Attribute']] == tree[i][0]['Value']):
                return decision_tree_classification(tree[i][1], row)

    else:
        maxStat = tree[0][1]
        maxClass = tree[0][0]
        for stat in tree:
            if stat[1] > maxStat:
                maxClass = stat[0]
                maxStat = stat[1]
        return maxClass

def stratified_kfold_validation_decision_tree(table):
    accuracy = 0.0
    true_positives_total = 0.0
    total_predictions = 0
    folds = create_stratified_kfolds(table)
    for i in range(10):
        test_set = folds[i]
        training_set = []
        for j in range(10):
            if (i != j):
                training_set += folds[j]

        tree = tdidt(training_set, {7:2, 4:2, -2:2, 9:2, 10:2, 11:2, 12:2}, 13)
        for index in range(len(test_set)):
            predictClass = decision_tree_classification(tree, test_set[index])
            actual = test_set[index][13]
            total_predictions += 1
            if (float(predictClass) > 0.0 and float(actual) > 0.0):
                true_positives_total += 1
            elif (float(predictClass) == 0.0 and float(actual) == 0.0):
                true_positives_total += 1

    accuracy = round((true_positives_total/total_predictions), 3)
    standardError = round((math.sqrt(accuracy*(1 - accuracy)/len(test_set))), 3)
    return accuracy, standardError





def create_stratified_kfolds_forest(table):
    disc_table = copy.deepcopy(table)
    disc_table.sort(key = operator.itemgetter(13))
    folds = [[] for _ in range(3)]
    index = 0
    for row in disc_table:
        folds[index].append(row)
        index = (index + 1) % 3
    return folds

def bootstrap(table):
    trainingSet, testSet = [], []
    for i in range(len(table)):
        trainingSet.append(table[random.randint(0, len(table) - 1)])
    for row in table:
        if row not in trainingSet:
            testSet.append(row)
    return trainingSet, testSet

def random_attr_subset(attributeIndexes, F):
    """ Randomly select F attributes to split on in random tree creation """
    attr = copy.deepcopy(attributeIndexes)
    randomIndexes = {}
    if (F < len(attr)):
        for i in range(F):
            index = random.randint(0, len(attr) - 1)
            randomIndexes[attr.keys()[index]] = attr[attr.keys()[index]]
            del attr[attr.keys()[index]]
    else:
        return attr
    return randomIndexes

def select_attribute_2(instances, F, att_indexes, class_index):
    """ Select attribute to split on using entropy of F unique random attributes """
    newIndexes = random_attr_subset(att_indexes, F)
    E = eNew(instances, newIndexes.keys()[0], class_index)
    att_index = newIndexes.keys()[0]
    for i in range(1, len(newIndexes)-1):
        temp_eNew = eNew(instances, newIndexes.keys()[i], class_index)
        if(temp_eNew < E):
            E = temp_eNew
            att_index = newIndexes.keys()[i]
    return att_index

def makeTree(instances, F, att_indexes, class_index):
    """ Creates a random tree based on F, used for making random forests """
    tree = []
    if (in_same_class(instances, class_index)):
        return partition_stats(instances, class_index)
    if (len(att_indexes) == 0):
        return partition_stats(instances, class_index)
    att_index = select_attribute_2(instances, F, att_indexes, class_index)
    partition_list = partition_instances(instances, att_index)
    if (len(partition_list) != att_indexes[att_index]):
        return partition_stats(instances, class_index)
    tree.append({'Attribute':att_index})
    att_indexes2 = copy.deepcopy(att_indexes)
    del att_indexes2[att_index]
    for i, v in partition_list.iteritems():
        tree.append([{'Value':i},makeTree(v, F, att_indexes2, class_index)])
    return tree

#forestTemp = forest, forest = ensemble
def build_rand_forest_ens(remainder_set, N, M, F):
    """ Returns list of random decision trees (forest) """
    forestTemp = []
    for i in range(N):
        train, test = bootstrap(remainder_set)
        tree = makeTree(train, F, {7:2, 4:2, -2:2, 9:2, 10:2, 11:2, 12:2}, 13)
        accuracy = test_standard_tree(tree, train)
        forestTemp.append([accuracy,tree])
    forestTemp.sort(key = operator.itemgetter(0))
    forest = []
    for i in range(N-1, N-M-1, -1):
        forest.append(forestTemp[i][1])
    return forest

def mode(classifiers):
    """ Returns a classifier based on simple majority voting """
    values, counts = [], []
    classifiers.sort()
    for classifier in classifiers:
        if classifier not in values:
            values.append(classifier)
            counts.append(1)
        else:
            counts[-1] += 1
    return values[counts.index(max(counts))]

def random_forest_classifier(row, rand_forest):
    """ Classifies a single row against each tree in a forest """
    classifiers = []
    for i in range(len(rand_forest)):
        classifiers.append(decision_tree_classification(rand_forest[i], row))
    return mode(classifiers)

def test_rand_forest_ens(test_set, rand_forest):
    """ Returns accuracy of the random forest classifier """
    total = float(len(test_set))
    totalRight = 0
    for i in range(len(test_set)):
        prediction = random_forest_classifier(test_set[i], rand_forest)
        actual = float(test_set[i][13])
        if (prediction == actual):
            totalRight += 1

    return totalRight/total

def test_standard_tree(tree, test_set):
    total = float(len(test_set))
    totalRight = 0
    for i in range(len(test_set)):
        prediction = decision_tree_classification(tree, test_set[i])
        actual = test_set[i][13]
        if prediction == actual:
            totalRight += 1

    return totalRight/total

def makeForest(table):
    remainderSet, testSet = [], []
    # create three folds
    FoldedTable = create_stratified_kfolds_forest(table)
    remainderSet += FoldedTable[0]
    remainderSet += FoldedTable[1]
    testSet += FoldedTable[2]
    forest = build_rand_forest_ens(remainderSet, 1, 1, 2)
    print "Forest Accuracy: ", test_rand_forest_ens(testSet, forest)
    tree = tdidt(remainderSet, {7:2, 4:2, -2:2, 9:2, 10:2, 11:2, 12:2}, 13)
    print "Standard Tree Accuracy: ", test_standard_tree(tree, testSet)






# deleted kickers, punters, long-snappers, nose tackles
# centers (C) changed to offensive centers (OC)
def pre_process(table):
    newTable = copy.deepcopy(table)
    for i in range(len(newTable) - 1, -1, -1):
        if table[i][3] == 'P' or table[i][3] == 'LS' or table[i][3] == 'K' or table[i][3] == "NT":
            print table[i][0]
        if newTable[i][3] == "C":
            newTable[i][3] = "OC"

    return newTable

def main():
    #combineTable = read_csv("combine.txt")

    # remove attributes that have too many null values
    #print_statistics(tempTable, [4, -1, 5, 6, 8, 9, 10], ["Weight (lbs)", "Height (in)", "40YD (sec)", "Shuttle (sec)", "Vert (in)", "Broad (in)", "Bench (reps)"])
    #newTable = pre_process(tempTable)
    #write_csv("processedTable", newTable)

    tempTable = read_csv("processedTable.txt")
    newTable1 = discretize_forty(tempTable)
    newTable2 = discretize_cont(newTable1, 4)
    newTable3 = discretize_cont(newTable2, -1)
    newTable4 = discretize_cont(newTable3, 7)
    newTable5 = discretize_cont(newTable4, 8)
    newTable6 = discretize_cont(newTable5, 9)
    newTable7 = discretize_cont(newTable6, 10)
    write_csv("splitPointTable_2", newTable7)

    treeTable = read_csv("splitPointTable.txt")
    for row in treeTable:
        if float(row[13]) > 0.0:
            row[13] = 1.0
        else:
            row[13] = 0.0
    #knnTable = read_csv("processedTable.txt")

    #makeForest(treeTable)
    #print stratified_kfold_validation_decision_tree(treeTable)
    #print stratified_kfold_validation_knn(knnTable) 

if __name__ == '__main__':
    main()
