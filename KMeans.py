import numpy as np
import random
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self):
        pass

    def KMeansDemo(self):
        centerList = [[0, 0], [30, 30], [30, -30], [-30, 30], [-30, -30]]
        sigmaList = []
        clustreNumList = []
        rangeList = []
        k = 5
        eleNum = 2
        for i in xrange(0, k):
            sigmaList.append(random.random() * 1.0)
            clustreNumList.append(np.int32(50.0 + random.random() * 50.0))
            rangeList.append(random.random() * 20.0)
        rawData = self._GenerateKClustre(2, centerList, sigmaList, clustreNumList, rangeList)
        mData = self._MergeData(rawData)
        clustre, centers = self.Kmeans(k, mData)
        plt.plot(centers[:, 0], centers[:, 1], 'rx', markersize=20)
        # plt.plot(clustre[0][:, 0], clustre[0][:, 1], 'ko', markersize=5, fillstyle='none')
        # plt.plot(clustre[1][:, 0], clustre[1][:, 1], 'yo', markersize=5, fillstyle='none')
        # plt.plot(clustre[2][:, 0], clustre[2][:, 1], 'go', markersize=5, fillstyle='none')
        # plt.plot(clustre[3][:, 0], clustre[3][:, 1], 'bo', markersize=5, fillstyle='none')
        # plt.plot(clustre[4][:, 0], clustre[4][:, 1], 'co', markersize=5, fillstyle='none')
        plt.plot(clustre[0][:, 0], clustre[0][:, 1], 'ko', fillstyle='none')
        plt.plot(clustre[1][:, 0], clustre[1][:, 1], 'yo', fillstyle='none')
        plt.plot(clustre[2][:, 0], clustre[2][:, 1], 'go', fillstyle='none')
        plt.plot(clustre[3][:, 0], clustre[3][:, 1], 'bo', fillstyle='none')
        plt.plot(clustre[4][:, 0], clustre[4][:, 1], 'co',  fillstyle='none')
        plt.show()

    def Kmeans(self, k, data):
        centers = self._SelectSeeds(data, k)
        iterCentroid = centers
        iterCentroidOld = np.zeros((iterCentroid.shape))
        clustreCenter = np.zeros((centers.shape[0], data.shape[1]))
        clustreNum = np.zeros((centers.shape[0],1))
        dataTag = np.zeros((data.shape[0],1), np.uint32)
        while np.linalg.norm(iterCentroid - iterCentroidOld) > 3:
            iterCentroidOld = iterCentroid
            for i in xrange(0, data.shape[0]):
                nearestCenter, nearestIndex = self._GetNearestClustreCenter(centers, data[i, :])
                dataTag[i] = nearestIndex
                clustreNum[nearestIndex] += 1
                clustreCenter[nearestIndex, :] += data[i, :]
            for i in xrange(0, iterCentroid.shape[0]):
                iterCentroid[i, :] = clustreCenter[i, :] / clustreNum[i]
        clustre = [np.zeros((0, data.shape[1])) for i in xrange(0,centers.shape[0])]
        for i in xrange(0, data.shape[0]):
            clustre[dataTag[i]] = np.vstack((clustre[dataTag[i]], data[i, :]))
        return clustre, centers

    def _MergeData(self, rawData):
        if len(rawData) == 0:
            return None
        mData = rawData[0]
        for i in xrange(1, len(rawData)):
            mData = np.concatenate((mData, rawData[i]), axis=0)
        return mData

    def _GenerateKClustre(self, eleNum, centerList, sigmaList, clustreNumList, rangeList):
        rawData = []
        for i in xrange(0, len(clustreNumList)):
            tmpData = np.zeros((clustreNumList[i], eleNum))
            center = centerList[i]
            sigma = sigmaList[i]
            for j in xrange(0, clustreNumList[i]):
                for ij in xrange(0, eleNum):
                    tmpData[j, ij] = center[ij] + random.gauss(0, sigma) * rangeList[i]
            rawData.append(tmpData)
        return rawData


    def _GetNearestClustreCenter(self, centers, datum):
        distMatrix = np.zeros((centers.shape[0], 1))
        for i in xrange(0, centers.shape[0]):
            distMatrix[i] = np.linalg.norm(datum - centers[i])
        nearestIndex = np.argmin(distMatrix)
        nearestCenter = centers[nearestIndex, :]
        return nearestCenter, nearestIndex

    def _SelectSeeds(self, data, k):
        #row vector
        seed = np.zeros((k, data.shape[1]))
        seed[0,:] = data[np.int32(np.random.random() * data.shape[0]), :]
        for i in xrange(1, k):
            seed[i, :] = self._GetFarestData(data, seed[0:i, :])
        return seed

    def _GetFarestData(self, data, curSeeds):
        distMatrix = np.zeros((data.shape[0],1))
        for i in xrange(0, data.shape[0]):
            nearestCenter, nearestIndex = self._GetNearestClustreCenter(curSeeds, data[i, :])
            distMatrix[i] = np.linalg.norm(data[i, :] - nearestCenter)
        farestIndex = np.argmax(distMatrix)
        newSeed = data[farestIndex, :]
        return newSeed





