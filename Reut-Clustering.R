#SECTION 1: data and pre-processing

library(RTextTools)
library(SnowballC)
library(tm)
library(XML)
library(e1071)
library(randomForest)
library(cluster)
library(fpc)
library(topicmodels)
library(class)

install.packages("/dcs/pg11/phil/reuters/tm.corpus.Reuters21578/", repos = NULL, type="source")
require(tm.corpus.Reuters21578)
data(Reuters21578)
test <- Reuters21578

Dati <- as.data.frame(matrix(0,21578, 3))
colnames(Dati) <- c("TOPICS","Topics","LEWISSPLIT")
for(i in 1:21578)
{
	doc <- test[[i]]
	Dati[i,1] <- LocalMetaData(doc)$TOPICS
	Dati[i,2] <- LocalMetaData(doc)$Topics[1]
	Dati[i,3] <- LocalMetaData(doc)$LEWISSPLIT
}


cllist <- as.integer(rownames(Dati[
			is.element(Dati$Topics, c("earn","acq","grain","trade"))
& is.element(Dati$LEWISSPLIT,c("TRAIN","TEST")),]))

Test <- as.integer(rownames(Dati[is.element(Dati$Topics, c("earn","acq",
"grain","trade")) & 
is.element(Dati$LEWISSPLIT,"TEST"),]))

Train <- as.integer(rownames(Dati[is.element(Dati$Topics, c("earn","acq",
"grain","trade")) & 
is.element(Dati$LEWISSPLIT,"TRAIN"),]))

Use <- test[cllist]

Use  <- tm_map(Use , as.PlainTextDocument)
Use  <- tm_map(Use , stripWhitespace)
Use  <- tm_map(Use , tolower)		
Use  <- tm_map(Use , removeWords, stopwords("english"))
Use  <- tm_map(Use , stemDocument)
Use  <- tm_map(Use , removePunctuation)
Use  <- tm_map(Use , removeNumbers)

dtm <- DocumentTermMatrix(Use)
dtm_tfxidf <- weightTfIdf(dtm)	#TF*IDF
DFterms <- as.data.frame(inspect(dtm_tfxidf))

words <-(as.matrix(inspect(removeSparseTerms(dtm_tfxidf, 0.95))))
veri <- factor(Dati[cllist,2])

w2 <- as.matrix(inspect(removeSparseTerms(dtm, 0.95)))

#SECTION 2: Clustering algorithms

d <- dist(words, method="euclidean") 

cl1 <- kmeans(words , 4, nstart = 500)	#KMEANS
K <- table(cl1$cluster, veri)
s1 <- silhouette(cl1$cl, d^2)
plot(s1, main="Silhouette K-Means")

clu1 <- pam(w2, 4) #PAM 
P<-table(clu1$cluster, veri)
si <- silhouette(clu1)
str <- str(si)
plot(si, main="Silhouette PAM")

clu2 <- hclust(d, method="ward") #HIER 
plot(clu2)
cl2 <- cutree(clu2, 4)
H <- table(cl2, veri)
si3 <- silhouette(cl2, d^2)
plot(si3, main="Silhouette Hierarchical")

#SECTION 3: Clustering evaluation

BH <- addmargins(H)
EH <- numeric(4)

for(i in 1:4)
{
	for(j in 1:4)
	{
		tmp <- 0 
		if(H[i,j]!=0)
			tmp <- H[i,j]/BH[i,5] * log(H[i,j]/BH[i,5],2) 
	}
	EH[i] <- BH[i,5]/BH[5,5]*sum(tmp)
}
-sum(EH)

BK <- addmargins(K)
EK <- numeric(4) 
for(i in 1:4)
{
	for(j in 1:4)
	{
		tmp <- 0 
		if(K[i,j]!=0)
			tmp <- K[i,j]/BK[i,5] * log(K[i,j]/BK[i,5],2) 
	}
	EK[i] <- BK[i,5]/BK[5,5]*sum(tmp)
}
-sum(EK)

BP <- addmargins(P)
EP <- numeric(4) 
for(i in 1:4)
{
	for(j in 1:4)
	{
		tmp <- 0 
		if(P[i,j]!=0)
			tmp <- P[i,j]/BP[i,5] * log(P[i,j]/BP[i,5],2) 
	}
	EP[i] <- BP[i,5]/BP[5,5]*sum(tmp)
}
-sum(EP)
