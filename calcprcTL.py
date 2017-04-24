import os,sys
import argparse
import shutil
import time
import re
import csv
import operator
from multiprocessing import Pool
from copy import deepcopy
#import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import metrics
from numba import jit

@jit
def findMinMaxInGT(annotations):
    min = 999999;
    max = 0;
    for line in annotations:
        lineSplits = line.split(";")
        #print lineSplits
        #print int(lineSplits[7])
        gtFrameNumber = int(lineSplits[7])

        if (max < gtFrameNumber ):
            max = gtFrameNumber
        if (min > gtFrameNumber):
            min = gtFrameNumber
    return min,max
@jit
def checkIfGtFrameExists(lookupNumber,annotations):
    for line in annotations:
        lineSplits = line.split(";")
        gtFrameNumber = int(lineSplits[7])
        #print("Checking frame: %s" % gtFrameNumber)
        if(gtFrameNumber == lookupNumber):
            return True
    return False
@jit
def sortFile(filePath):
    fileToSort = open(filePath,'rU')

    reader = csv.reader(fileToSort, delimiter=";")
    sortedList = sorted(reader, key=operator.itemgetter(0), reverse=False)
    fileToSort.close()

    outputFile = open(filePath, 'w')
    writer = csv.writer(outputFile, delimiter=";",lineterminator="\n")
    writer.writerows(sortedList)
    outputFile.close()

def stripFile(filepath):
    clean_lines = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        clean_lines = [l.strip() for l in lines if l.strip()]

    with open(filepath, "w") as f:
        f.writelines('\n'.join(clean_lines))
@jit
def convertDetectionToArray(detections):
    dictDetections = {}

    counter = 0 # Lines in file start from 1
    for line in detections:
        dtLineSplits = line.split(";")
        testSeqNameUnSegmented = dtLineSplits[0]
        testSeqNumbers = (re.findall(r'\d+',testSeqNameUnSegmented))

        dtFrameNumber = int(testSeqNumbers[1])
        dictDetections.setdefault(dtFrameNumber, [])
        dictDetections[dtFrameNumber].append(counter)
        counter = counter + 1
    return dictDetections
@jit
def convertGtToArray(annotations):
    dictAnnotations = {}

    counter = 0 # Lines in file start 1

    for line in annotations:
        lineSplits = line.split(";")
        gtFrameNumber = int(lineSplits[7])
        dictAnnotations.setdefault(gtFrameNumber, [])
        dictAnnotations[gtFrameNumber].append(counter)
        counter = counter + 1
    return dictAnnotations
@jit
def convertGtToRecognitionArray(annotations,args):
    dictAnnotationsTP = {}
    dictAnnotationsTN = {}

    counter = 0 # Lines in file start 1

    for line in annotations:
        lineSplits = line.split(";")
        gtType = lineSplits[1]
        gtFrameNumber = int(lineSplits[7])

        if( gtType == args.classType):
            dictAnnotationsTP.setdefault(gtFrameNumber, [])
            dictAnnotationsTP[gtFrameNumber].append(counter)
        else:
            dictAnnotationsTN.setdefault(gtFrameNumber, [])
            dictAnnotationsTN[gtFrameNumber].append(counter)

        counter = counter + 1
    return dictAnnotationsTP,dictAnnotationsTN

@jit
def lengthOfMetric(workingDict):
    length = 0
    for key,value in workingDict.iteritems():
        length = length + len(value)
    return length
@jit
def pascalCriteria(pascalVar,gtX1,gtX2,gtY1,gtY2,dtX1,dtX2,dtY1,dtY2):
    # Compute intersection
    left = max(gtX1, dtX1)
    right = min(gtX2, dtX2)
    top = max(gtY1,dtY1)
    bottom = min(gtY2,dtY2)
    #print(left,right,top,bottom)
    if left < right and top < bottom:
        intersectionArea = (right-left)*(bottom-top)
    else:
        intersectionArea = 0
    #print("Intersections area is: %s" % intersectionArea)
    unionArea = ( ((gtX2-gtX1) * (gtY2-gtY1)) + ((dtX2-dtX1) * (dtY2-dtY1)) )-intersectionArea

    pascalMeasure = float(intersectionArea)/float(unionArea)
    #print("Intersection area: %s Union Area: %s Overlap: %s" % (intersectionArea, unionArea, pascalMeasure))
    if(pascalMeasure >= pascalVar):
        return True
    return False


def compareDtToGt(dictGt,dictDt,pascal,annotations, detections,args):
    if(args.classType != "detector"):
        dictGt = {}
        dictGt, dictAnnotationsTN = convertGtToRecognitionArray(annotations,args)
        workingDictTN = deepcopy(dictAnnotationsTN)

    maxValInGt = max(dictGt.keys(), key=int)
    minValInGt = min(dictGt.keys(), key=int)

    workingDictDt = deepcopy(dictDt)
    workingDictGt = deepcopy(dictGt)

    workingDictTP = {}
    workingDictFP = deepcopy(dictDt)
    workingDictFN = deepcopy(dictGt)
    goFpCounter = 0
    warningFpCounter = 0
    stopFpCounter = 0
    goLeftFpCounter = 0
    warningLeftFpCounter = 0
    stopLeftFpCounter = 0
    goForwardFpCounter = 0

    for i in workingDictDt:
        #print("Examening frame-%s" % i)
        if i in workingDictGt.keys():
            #print("Frame-%s has %s detections and %s annotations" % (i,len(dictDt[i]),len(dictGt[i])))
            for dtVal in workingDictDt[i]:
                if( i >= minValInGt and i <= maxValInGt ):

                    dtLineSplits = detections[dtVal].split(";")

                    testSeqNameUnSegmented = dtLineSplits[0]
                    testSeqNumbers = (re.findall(r'\d+',testSeqNameUnSegmented))
                    dtFrameNumber = int(testSeqNumbers[1])

                    dtX1 = int(dtLineSplits[1]) # Upper left corner X
                    dtY1 = int(dtLineSplits[2]) # Upper left corner Y
                    dtX2 = int(dtLineSplits[3]) # Lower right corner X
                    dtY2 = int(dtLineSplits[4]) # Lower right corner Y
                    dtScore = float(dtLineSplits[5]) # Lower right corner Y
                    hitBool = False
                    hitsScore = 0
                    for gtVal in workingDictGt[i]:
                        # Compare detections with annotations
                        line = annotations[gtVal]
                        lineSplits = line.split(";")
                        gtFrameNumber = int(lineSplits[7])

                        gtType = lineSplits[1]

                        if( gtType == args.classType or args.classType == "detector" ):
                            gtX1 = int(lineSplits[2]) # Upper left corner X
                            gtY1 = int(lineSplits[3]) # Upper left corner Y
                            gtX2 = int(lineSplits[4]) # Lower right corner X
                            gtY2 = int(lineSplits[5]) # Lower right corner Y

                            if(pascalCriteria(args.pascal,gtX1,gtX2,gtY1,gtY2,dtX1,dtX2,dtY1,dtY2)):
                                if( dtScore > hitsScore):
                                    hitBool = True
                                    hitsScore = dtScore
                                    workingGtVal = gtVal
                                    workingDtVal = dtVal

                    if(hitBool):
                        workingDictTP.setdefault(i, [])
                        workingDictTP[i].append(workingDtVal)

                        workingDictFP[i].remove(workingDtVal)
                        #print ("workingDictFN: %s workingGtVal: %s" % (workingDictFN[i],workingGtVal) )
                        workingDictFN[i].remove(workingGtVal)
                        workingDictGt[i].remove(workingGtVal)

                        hitBool = False

                    elif(hitBool == False and args.classType != "detector" ):
                        hitFp = False
                        if i in workingDictTN.keys():
                            for gtValA in workingDictTN[i]:

                                # Compare detections with annotations
                                line = annotations[gtValA]
                                #print line
                                lineSplits = line.split(";")
                                gtFrameNumber = int(lineSplits[7])

                                gtType = lineSplits[1]
                                if ( gtType != args.classType  and args.classType != "detector" ):
                                    gtX1 = int(lineSplits[2]) # Upper left corner X
                                    gtY1 = int(lineSplits[3]) # Upper left corner Y
                                    gtX2 = int(lineSplits[4]) # Lower right corner X
                                    gtY2 = int(lineSplits[5]) # Lower right corner Y

                                    if(pascalCriteria(args.pascal,gtX1,gtX2,gtY1,gtY2,dtX1,dtX2,dtY1,dtY2)):
                                        hitFp = True
                                        workingDictFP[i].append(dtVal) # Append wrong detectiong to False positives
                                        workingDictTN[i].remove(gtValA) # Remove from true negative dict
                                        if(gtType == "go" and gtType != args.classType ):
                                            goFpCounter = goFpCounter + 1
                                        if(gtType == "warning" and gtType != args.classType ):
                                            warningFpCounter = warningFpCounter + 1
                                        if(gtType == "stop" and gtType != args.classType ):
                                            stopFpCounter = stopFpCounter + 1
                                        if(gtType == "goLeft" and gtType != args.classType ):
                                            goLeftFpCounter = goLeftFpCounter + 1
                                        if(gtType == "warningLeft" and gtType != args.classType ):
                                            warningLeftFpCounter = warningLeftFpCounter + 1
                                        if(gtType == "stopLeft" and gtType != args.classType ):
                                            stopLeftFpCounter = stopLeftFpCounter + 1
                                        if(gtType == "goForward" and gtType != args.classType ):
                                            goForwardFpCounter = goForwardFpCounter + 1

    lengthsTP = lengthOfMetric(workingDictTP)
    lengthsFP = lengthOfMetric(workingDictFP)
    lengthsFN = lengthOfMetric(workingDictFN)
    resultsString = []
    myHeaderStr = "Evaluation results for '%s' class '%s'" % (args.detections, args.classType)

    resultsString.append(myHeaderStr)
    if(args.classType != "detector" ):
        lengthsTN = lengthOfMetric(workingDictTN)
        myResults = "Number of True Positives: %s\nNumber of False Positives: %s\nNumber of False Negatives: %s\nNumber of True Negatives: %s" % (lengthsTP,lengthsFP,lengthsFN,lengthsTN)
        resultsString.append(myResults)
        resultsString.append("\nFalse Positives Analysis:\nNumber of 'go':%s\nNumber of 'warning':%s\nNumber of 'stop':%s\nNumber of 'goLeft':%s\nNumber of 'warningLeft':%s\nNumber of 'stopLeft':%s\nNumber of 'goForward':%s\n" % (goFpCounter,warningFpCounter,stopFpCounter,goLeftFpCounter,warningLeftFpCounter,stopLeftFpCounter,goForwardFpCounter))
    else:
        myResults = "Number of True Positives: %s\nNumber of False Positives: %s\nNumber of False Negatives: %s" % (lengthsTP,lengthsFP,lengthsFN)
        resultsString.append(myResults)
    print(myResults)


    score, target = sortScoreTarget(workingDictTP,workingDictFP,workingDictFN,detections)

    return score, target, resultsString

def sortScoreTarget(workingDictTP,workingDictFP,workingDictFN,detections):
    dtTPTarget = "0"
    dtFPTarget = "1"
    dtFNTarget = "2"
    dtFNScore = "NaN"
    score = []
    target = []

    for i in workingDictTP:
        for val in workingDictTP[i]:
            dtLineSplits = detections[val].split(";")
            dtScoreSplits = dtLineSplits[5].split("\n")
            dtScore = dtScoreSplits[0]
            score.append(dtScore)
            target.append(dtTPTarget)

    for i in workingDictFP:
        for val in workingDictFP[i]:
            dtLineSplits = detections[val].split(";")
            dtScoreSplits = dtLineSplits[5].split("\n")
            dtScore = dtScoreSplits[0]
            score.append(dtScore)
            target.append(dtFPTarget)

    for i in workingDictFN:
        for val in workingDictFN[i]:
            dtLineSplits = detections[val].split(";")
            dtScoreSplits = dtLineSplits[5].split("\n")
            dtScore = dtScoreSplits[0]
            score.append(dtFNScore)
            target.append(dtFNTarget)
    return score, target

@jit
def calcPRC(score,target,args,resultsString):
    #numberOfBins = 2001
    numberOfBins = 200

    def greaterThanZero(element):
        return element >= 0

    if(args.printToFile):
        #scoresFile =  open(os.path.abspath("%s-scores.txt"%args.classType),'w')
        #targetsFile =  open(os.path.abspath("%s-targets.txt"%args.classType),'w')
        scoresFile =  open(os.path.abspath("results/%s-%s/%s-%s-scores.txt" % (str(args.classType),str(args.printToFile),str(args.classType),str(args.printToFile)) ),'w')
        targetsFile =  open(os.path.abspath("results/%s-%s/%s-%s-targets.txt" % (str(args.classType),str(args.printToFile),args.classType,args.printToFile) ),'w')
        for item in score:
            scoresFile.write("%s\n" % item)
        for item in target:
            targetsFile.write("%s\n" % item)

    #print([i for i in score if i != '-1' or i != 'NaN'])
    thresScoreValsList = [i for i in score if i != '-1' and i != 'NaN']
    minThresVal = float(min( thresScoreValsList ))
    maxThresVal = float(max( thresScoreValsList ))

    print("Min thres val: %s Max thres val: %s" % (minThresVal, maxThresVal))

    thresEvenlyBinsList = []
    thresIter = 0

    thresValsList = []
    thresIter = 0
    sortedScores = deepcopy(score)
    sortedScores = [i for i in sortedScores if i != '-1' and i != 'NaN']
    sortedScores.sort(reverse=True)
    thresBinIntervals = len(sortedScores)/numberOfBins
    #print("The bin intervals: %s" % thresBinIntervals)

    while (thresIter <= numberOfBins):
        if (thresIter == numberOfBins):
            thresValsList.append(float(min(sortedScores)))
        else:
            thresValsList.append(sortedScores[thresBinIntervals*thresIter])
        thresIter = thresIter + 1

    precision = []
    recall = []
    fns = 0
    firstRun = True
    smallChanges = 0
    prevRecall = 999999
    prevPrecision = 999999
    prevThreshold = 999999.0

    for thres in thresValsList:
        workingPrecision, workingRecall = testJitFunction(thres,score,target,firstRun,fns)

        precision.append(workingPrecision)
        recall.append(workingRecall)


        #print("Thres: %.6f Precision: %.6f Recall: %.6f DiffPrecision: %.6f DiffRecall: %.6f DiffThres: %.6f smallChanges: %s" % (float(thres), float(workingPrecision), float(workingRecall), float(diffPrecision), float(diffRecall), float(diffThres), smallChanges ) )
        #if smallChanges > 200:
            #print("The changes are small now. Breaking")
            #break

    hitBottomRecall = workingRecall
    hitBottomPrecision = 0
    print("Highest precision: %s" % max(precision))
    print("Highest recall: %s" % max(recall))

    recall.append(hitBottomRecall)
    precision.append(hitBottomPrecision)

    if(args.plot):
        plotPRC(precision,recall,args,resultsString)
@jit
def testJitFunction(thres,score,target,firstRun,fns):
    counter = 0
    tps = 0
    fps = 0

    for i in score:
        if float(i) >= float(thres) and i != '-1' and i != 'NaN':
            if (target[counter] == '0'): # True positives
                tps = tps + 1
            if (target[counter] == '1'): # False positives
                fps = fps + 1
        if(i == '-1' or i == 'NaN' and firstRun == True):
            if (target[counter] == '2'): # False negatives
                fns = fns + 1

        counter = counter + 1 # Counter to identify line number in score and target list

    firstRun = False
    #print("Thres: %s TPs: %s FPs %s FNs %s" % (thres, tps, fps, fns) )
    if fps == 0:
        workingPrecision = 1
    else:
        workingPrecision = float(tps)/(float(tps) + float(fps))

    if(fns==0):
        workingRecall = 1
    else:
        workingRecall = float(tps)/(float(tps) + float(fns))

    return workingPrecision, workingRecall

def plotPRC(precision,recall,args,resultsString):
    #saveToFileName = "%s-%s.png" % (args.classType,args.printToFile)
    saveToFileName = "results/%s-%s/%s-%s.png" % (str(args.classType),str(args.printToFile),args.classType,args.printToFile)
    print("Plotting and saving to file: %s" % saveToFileName)
    plt.clf()

    #matrixRePr = np.column_stack((recall, precision))
    #sortedMatrixRePr = sorted(matrixRePr,key=operator.itemgetter(0),reverse=True)
    #recall = recall.sort(reverse=False)
    #precision = precision.sort(reverse=True)

    #auc = metrics.auc(recall, precision)*100
    recall, precision = (list(t) for t in zip(*sorted(zip(recall, precision),reverse=True)))
    auc = metrics.auc(recall, precision)*100

    labelLegend = args.legend + "-" + args.classType+ ('(AUC={0:0.2f}%)'.format(auc))

    plt.plot(recall, precision, label=labelLegend,linewidth=5.0)
    #plt.plot(recall, precision, 'o')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(args.title)
    plt.legend(loc="upper right")

    #myFilenameStr = "%s.txt" % args.classType
    #print myFilenameStr
    myAuc = ('Area under Curve(AUC): {0:0.2f} %'.format(auc))
    resultsString.append(myAuc)
    if(args.printToFile):
        plt.savefig(saveToFileName)
        theFile = open("results/%s-%s/%s-%s-results.txt" % ( str(args.classType),str(args.printToFile),str(args.classType),str(args.printToFile)), 'w')
        for row in resultsString:
            print row
            theFile.write("%s\n" % str(row))
        theFile.close()

        thePrecisionFile = open("results/%s-%s/%s-%s-Precision.txt" % ( str(args.classType),str(args.printToFile),str(args.classType),str(args.printToFile)), 'w')
        for row in precision:
            #print row
            thePrecisionFile.write("%s\n" % str(row))
        thePrecisionFile.close()

        theRecallFile = open("results/%s-%s/%s-%s-Recall.txt" % ( str(args.classType),str(args.printToFile),str(args.classType),str(args.printToFile)), 'w')
        for row in recall:
            #print row
            theRecallFile.write("%s\n" % str(row))
        theRecallFile.close()
    #if(args.plot):
        #plt.show()


def main(args):
    if not os.path.isfile(args.groundTruth):
        print("ERROR: The annotation file: %s does not exist." % args.groundTruth)
        exit()
    if not os.path.isfile(args.detections):
        print("ERROR: The detection file: %s does not exist." % args.detections)
        exit()
    directory = "results/%s-%s/"% ( str(args.classType),str(args.printToFile))
    if not os.path.exists(directory):
        os.makedirs(directory)
    # We sort input files in decending order, and strip them for any blank lines.
    t0 = time.time()
    stripFile(args.groundTruth)
    stripFile(args.detections)
    sortFile(args.groundTruth)
    sortFile(args.detections)
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of sorting all input files: %f" %totalExecTime)

    ## Read annotations
    t0 = time.time()
    annotationFile = open(os.path.abspath(args.groundTruth),'r')
    header = annotationFile.readline() #Discard the header line
    annotations = annotationFile.readlines()
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of reading in annotations: %f" %totalExecTime)

    ## Read detections
    t0 = time.time()
    detectionFile = open(os.path.abspath(args.detections))
    detections = detectionFile.readlines()
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of reading in detections: %f" %totalExecTime)

    ## We find the min and max annotations to skip looking for matches before and after the first and last annotations.
    t0 = time.time()
    min,max = findMinMaxInGT(annotations)
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of finding min and max values in annotations: %f" %totalExecTime)

    t0 = time.time()
    dictDetections = convertDetectionToArray(detections)
    dictAnnotations = convertGtToArray(annotations)
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of converting dt and gt to dicts: %f" %totalExecTime)

    t0 = time.time()
    score,target,resultsString = compareDtToGt(dictAnnotations,dictDetections,args.pascal,annotations,detections,args)
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of comparing dt and gt dicts: %f" %totalExecTime)

    t0 = time.time()
    calcPRC(score,target,args,resultsString)
    t1 = time.time()
    totalExecTime = t1-t0
    print("Execution time of calculating PR-curve: %f" %totalExecTime)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and plot the precision-recall curve based on a detection and ground truth file.', epilog='This program will create a several several new files, however the \'results.csv\' and the plot are the most interesting, as it contains the evalutation results. Example of use: python calcprcTL.py -gt frameAnnotationsBULB.csv -d go.csv -p 0.5 -o -ptf prcPlot -c "go" -l "AFC"')
    parser.add_argument('-gt','--groundTruth', metavar='annotations.csv', type=str, help='Path to the csv file containing the annotations/ground truths.')
    parser.add_argument('-d','--detections', metavar='detections.csv', type=str, help='Path to the csv file containing the detections.')
    parser.add_argument('-c','--classType', metavar="detector", default="detector", type=str, help='Define recognition type (go,warning,stop,goLeft,warningLeft,stopLeft,goForward). Default: detector(Includes all classes).')
    parser.add_argument('-p','--pascal', metavar=0.5, type=float, default=0.5, help='Define the Pascal overlap criteria (By default: 0.5)')
    parser.add_argument('-t','--title', default='PRC Plot', metavar="\"PRC Plot\"", help='Title to put on the plot')
    parser.add_argument('-l','--legend', default='PRC Plot', metavar="\"PRC Legend\"", help='Legend to put on the plot')
    parser.add_argument('-o','--plot', action='store_true', help='Show plot of the computed PR-curve')
    parser.add_argument('-ptf','--printToFile', default='prcPlot.png',  metavar='prcPlot.png', type=str, help='Print the scores and targets to file.')

    args = parser.parse_args()

    main(args)
    #main()
