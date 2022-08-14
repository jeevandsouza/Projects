import random as rd
import pandas as pd
import re
import math
import wget
#wget is a URL network downloader that can work in the background, and it helps in downloading files directly from the main 
#server. In Python, this task is done by using the wget module.

class KMeansClustering:
    #writing a constructor to get the name of file or url
    def __init__(self,URLorFile):
        self.df = open(URLorFile,'r') # open the file in readonly mode
        self.getTweets = list(self.df) # storing it as a list inside a constructor
    
        #Next step is preprocessing data
#In preprocessing we have to remove tweetid and stamp
    def TweetPreprocessing(self):
        self.tweetArray = []
        for itr in range(len(self.getTweets)):
            self.getTweets[itr] = self.getTweets[itr].lower() # to lowercase
            self.getTweets[itr] = self.getTweets[itr][50:] # all tweetid timestamp length is equal to 50 so splice and take the result
            self.getTweets[itr] = self.getTweets[itr].replace("#","") #Replacing hastag with empty space
            self.getTweets[itr] = " ".join(filter(lambda y: y[0] != '@', self.getTweets[itr].split()))  #getting rid of words starting with @
            self.getTweets[itr] = self.getTweets[itr].strip() #remove all the whitespaces
            self.getTweets[itr] = re.sub(r"http\S+", "", self.getTweets[itr])  #remove all URL's if present
        #we have to remove if any video, gif etc is present
            arrLen = len(self.getTweets[itr]) # get the array length of the tweet, loop through it
            if arrLen > 0:
                if self.getTweets[itr][len(self.getTweets[itr]) - 1] == ':':
                    self.getTweets[itr] = self.getTweets[itr][:len(self.getTweets[itr]) - 1]
                self.getTweets[itr] = re.sub(r'[^\w\s]', '', self.getTweets[itr])
                self.getTweets[itr] = self.getTweets[itr].strip() #remove the video
                self.getTweets[itr] = " ".join(self.getTweets[itr].split())
                self.tweetArray.append(self.getTweets[itr].split(" ")) # append the array result

    def CalculateJaccardianDist(self,dist1,dist2):
        #Jaccardian distance is calculated using 1 - (A and b)/(A or B)
        # we can use intersetion and union functions present also use set to not allow duplicates
        union_set = set().union(dist1,dist2) # get the AUB result
        intersetion_set = set(dist1).intersection(set(dist2)) #get the A and B result
        dist3 = 1- (len(intersetion_set)/len(union_set))
        return dist3 # return the calculated distance

    def SumOfSquareErrors(self,dist):
    # we need to calculate sum error, loop through the cluster and within each cluster a point and calculate total error
        sumError = 0
        for i in range(len(dist)):   # each cluster
            for j in range(len(dist[i])): # point inside each cluster
                sumError = sumError+(dist[i][j][1]**2)  # sum up squared errors
        return sumError # return the error

    def PointAllocation(self):
    #using a dictionary to store the distance and centroids that they belong to 
        ClusterDict = dict() # inititalize the dictionary
        for i in range(len(self.tweetArray)): # loop through every tweet
            min_dist = math.inf # keeping the max distance so as to compare this with centroid distance
            index = -1
            for j in range(len(self.centroids)):
            # if the point itself is the centroid break and append it to the list
                getDist = self.CalculateJaccardianDist(self.tweetArray[i],self.centroids[j])
                if(self.tweetArray[i] == self.centroids[j]):
                    min_dist = 0 # distance is ofcourse 0
                    index = j 
                    break
            #calculate using jaccardian distance
                # getDist = self.CalculateJaccardianDist(self.tweetArray[i],self.centroids[j])
            # check with min dist if yes save it else continue
                if(getDist < min_dist):
                    min_dist = getDist
                    index = j
            ClusterDict.setdefault(index, []).append([self.tweetArray[i]])
            idx = len(ClusterDict.setdefault(index, [])) - 1
            ClusterDict.setdefault(index, [])[idx].append(min_dist) # append the distanc of the point to the cluster 
        return ClusterDict

    def RevisitAndChange(self,givenCluster):
        store = []
    # loop through the cluster
        for i in range(len(givenCluster)):
            min_dist = math.inf
            index = -1
            for j in range(len(givenCluster[i])):
                lenDist = 0
            #calculate the jaccaradian distance again
                for k in range(len(givenCluster[i])):
                    if (j != k):
                        l = self.CalculateJaccardianDist(givenCluster[i][j][0],givenCluster[i][k][0])
                        lenDist = lenDist+l # add to the distance
                if(lenDist < min_dist): # compare the min with calculated and store len and index
                    lenDist = min_dist
                    index = j
            store.append(givenCluster[i][index][0]) # append it to the store
        return store

    def Merging(self):
    # if past and present centroid length are not same return false
        if len(self.centroids) != len(self.past_centroid):
            return False
        for i in range(len(self.past_centroid)):
            if(" ".join(self.past_centroid[i]) != " ".join(self.centroids[i])):# join the i th centroid and check if equal
                return False
            return True

    #definition of k means algo
    def K_Means_Algo(self,k_val,iterations=100): # giving default iterations as 100
        self.centroids = []
        index = rd.sample(range(len(self.tweetArray)), k_val)
        for i in index:
            self.centroids.append(self.tweetArray[i])
        count = 0 # keep a count for iterations
        self.past_centroid = [] # to track past centroids
        while((self.Merging() == False) and (count < iterations)):
            cluster = self.PointAllocation() # iterate thorugh points and assign them to the cluster
            self.past_centroid = self.centroids # store it in another array
            self.centroids = self.RevisitAndChange(cluster) # visit the cluster and compute 
            count=count+1 # increment the counter
        SSE_error = self.SumOfSquareErrors(cluster)
        return cluster, SSE_error

    def CalculateandPrint(self):
        df = pd.DataFrame(columns=['K', 'SSE', 'ClusterSize']) # use dataframe to store k value, sse and cluster size
        pd.set_option('display.max_colwidth', 0)
        total_iterations = 5 # total iterations to be run
        k = 5 # It is generally used as the default k value
        for i in range(total_iterations):
            print("clustering for k = %s"%(k))
            clusters, error = KMeansClustering.K_Means_Algo(k) # call k means to compute the alogrithm
            Total_clusters = []
            for i in range(len(clusters)):
                print("Cluster %s -> %s tweets"%(i + 1, len(clusters[i])))
                Total_clusters.append(len(clusters[i]))
            print("SSE -> %s"%(error)+'\n')# print the sse error
            result = {' K': k,'SSE': error,'Cluster size': Total_clusters } # store k value,sse and cluster size
            df = df.append(result, ignore_index = True)
            k = k + 10 # increment the k value
        df.to_excel("table.xlsx", index=False)# store it to an excel file
        

if __name__ == "__main__":
    my_URL = 'https://github.com/rakshithamahesh/K-Means-Clustering/blob/main/foxnewshealth.txt'
    #my_URL = 'C:\Users\jeeva\Downloads\foxnewshealth.txt'
    get_Data = wget.download(my_URL)
    KMeansClustering = KMeansClustering(get_Data)
    KMeansClustering.TweetPreprocessing()
    KMeansClustering.CalculateandPrint()