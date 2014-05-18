#SECTION 1: Files and pre-processing

library(tm)
library(e1071)
library(randomForest)
library(cluster)
library(fpc)
library(topicmodels)
library(ggplot2)
library(reshape)

install.packages("/dcs/pg11/phil/reuters/tm.corpus.Reuters21578/", repos = NULL, type="source")
data(Reuters21578)
test<- Reuters21578

Dati <- as.data.frame(matrix(0,21578, 12))
colnames(Dati)[c(1,2,12)] <- c("TOPICS","Topics","LEWISSPLIT")
for(i in 1:21578)
{
	doc <- test[[i]]
	Dati[i,1] <- LocalMetaData(doc)$TOPICS
	n<-length(LocalMetaData(doc)$Topics)
	Dati[i,2:(min(12,(n+1)))] <- LocalMetaData(doc)$Topics[1:(min(11,n))]
	Dati[i,12] <- LocalMetaData(doc)$LEWISSPLIT
}

#Choose most frequent class
sub <- Dati[is.element(Dati$Topics, c("crude","earn","acq",
"money-fx","grain","trade","interest","ship","wheat","corn"))& Dati$V3==0, 2]
table(sub)

list <- as.integer(rownames(Dati[
			is.element(Dati$Topics, c("crude","earn","acq",
"money-fx","grain","trade","interest","ship","wheat","corn")) & 
(is.element(Dati$LEWISSPLIT, c("TEST","TRAINING-SET","TRAIN"))),]))

Test <- as.integer(rownames(Dati[is.element(Dati$Topics, c("crude","earn","acq",
"money-fx","grain","trade","interest","ship","wheat","corn")) & 
is.element(Dati$LEWISSPLIT,"TEST"),]))

Train <- as.integer(rownames(Dati[is.element(Dati$Topics, c("crude","earn","acq",
"money-fx","grain","trade","interest","ship","wheat","corn")) & 
is.element(Dati$LEWISSPLIT,"TRAIN"),]))

Use <- test[list]

Use  <- tm_map(Use , as.PlainTextDocument)
Use  <- tm_map(Use , stripWhitespace)
Use  <- tm_map(Use , tolower)		
Use  <- tm_map(Use , removeWords, stopwords("english"))
Use  <- tm_map(Use , stemDocument)
Use  <- tm_map(Use , removePunctuation)
Use  <- tm_map(Use , removeNumbers)

#SECTION 2: Classification

dtm <- DocumentTermMatrix(Use)
dtm_tfxidf <- weightTfIdf(dtm)	#TF*IDF
class <- as.factor(Dati[Train,2])

trials <- seq(0.45,0.95,0.05)
AB <-  numeric(11)
AS <-  numeric(11)
AR <-  numeric(11)
ARtest <- numeric(11)
AStest <- numeric(11)

for(i in 1:11)
{
	words <- as.data.frame(inspect(removeSparseTerms(dtm_tfxidf, trials[i])))


	NB <- naiveBayes(words[1:length(Train),], class, laplace = 1)
	B <- table(as.factor(Dati[Train,2]), predict(NB,words[1:length(Train),]))
	AB[i] <- sum(diag(B))/sum(B)

	SVM <- svm(words[1:length(Train),], class, cost=10, gamma=0.1, kernel ="linear")
	S <- table(as.factor(Dati[Train,2]), predict(SVM,words[1:length(Train),]))
	Stest <- table(as.factor(Dati[Test,2]), predict(SVM,words[(length(Train)+1):length(list),]))
	AS[i] <- sum(diag(S))/sum(S)
	AStest[i] <- sum(diag(Stest))/sum(Stest)

	RF <- randomForest(words[1:length(Train),], class)
	R <- table(as.factor(Dati[Train,2]), predict(RF,words[1:length(Train),]))
 	AR[i] <- sum(diag(R))/sum(R)
	Rtest <- table(as.factor(Dati[Test,2]), predict(RF,words[(length(Train)+1):length(list),]))
	ARtest[i] <- sum(diag(Rtest))/sum(Rtest)
	
}


data <- data.frame(sparsity= trials, NaiveBayes = AB, SVM = AS, RandomForest = AR )
Molten <- melt(data, id.vars = "sparsity")
ggplot(Molten, aes(x = sparsity, y = value, colour = variable)) + geom_line() +  ylab("Accuracy") +
  ggtitle("Accuracy vs. Sparsity plot - Train data")

data <- data.frame(sparsity= trials, SVM = AStest, RandomForest = ARtest )
Molten <- melt(data, id.vars = "sparsity")
ggplot(Molten, aes(x = sparsity, y = value, colour = variable)) + geom_line() +  ylab("Accuracy") +
  ggtitle("Accuracy vs. Sparsity plot - Test data")

Mat_NB <- B
INB <- matrix(0,10,3)
colnames(INB) <- c("Precision", "Recall", "F-Measure")
rownames(INB) <- c("acq", "corn","crude","earn","grain","interest",
"money-fx","ship","trade","wheat")
for(i in 1:10)
{
	INB[i,1] <- precision <- Mat_NB[i,i] / (sum(Mat_NB[,i]))
	INB[i,2] <- recall <- Mat_NB[i,i] / (sum(Mat_NB[i,]))
	INB[i,3] <- (2*INB[i,1]*INB[i,2])/(INB[i,1] + INB[i,2])
}

Mat_SVM <- S
colnames(SVM) <- c("Precision", "Recall", "F-Measure")
rownames(SVM) <- c("acq", "corn","crude","earn","grain","interest",
"money-fx","ship","trade","wheat")
for(i in 1:10)
{
	SVM[i,1] <- precision <- Mat_SVM[i,i] / (sum(Mat_SVM[,i]))
	SVM[i,2] <- recall <- Mat_SVM[i,i] / (sum(Mat_SVM[i,]))
	SVM[i,3] <- (2*SVM[i,1]*SVM[i,2])/(SVM[i,1] + SVM[i,2])
}

Mat_Rfa <- R
Rfa <- matrix(0,10,3)
colnames(Rfa) <- c("Precision", "Recall", "F-Measure")
rownames(Rfa) <- c("acq", "corn","crude","earn","grain","interest",
"money-fx","ship","trade","wheat")
for(i in 1:10)
{
	Rfa[i,1] <- precision <- Mat_Rfa[i,i] / (sum(Mat_Rfa[,i]))
	Rfa[i,2] <- recall <- Mat_Rfa[i,i] / (sum(Mat_Rfa[i,]))
	Rfa[i,3] <- (2*Rfa[i,1]*Rfa[i,2])/(Rfa[i,1] + Rfa[i,2])
}

Mat_Rfe <- Rtest
Rfe <- matrix(0,10,3)
colnames(Rfe) <- c("Precision", "Recall", "F-Measure")
rownames(Rfe) <- c("acq", "corn","crude","earn","grain","interest",
"money-fx","ship","trade","wheat")
for(i in 1:10)
{
	Rfe[i,1] <- precision <- Mat_Rfe[i,i] / (sum(Mat_Rfe[,i]))
	Rfe[i,2] <- recall <- Mat_Rfe[i,i] / (sum(Mat_Rfe[i,]))
	Rfe[i,3] <- (2*Rfe[i,1]*Rfe[i,2])/(Rfe[i,1] + Rfe[i,2])
}

round(cbind(INB, SVM, Rfa),3)

RMacro_NB <- mean(INB[,2])
PMacro_NB <- mean(INB[,1])
RMacro_SVM <- mean(SVM[,2])
PMacro_SVM <- mean(SVM[,1])
RMacro_RF <- mean(Rfa[,2])
PMacro_RF <- mean(Rfa[,1])

2*RMacro_NB*PMacro_NB/(RMacro_NB + PMacro_NB)
2*RMacro_RF*PMacro_RF/(RMacro_RF + PMacro_RF)
2*RMacro_SVM*PMacro_SVM/(RMacro_SVM + PMacro_SVM)

Mat_NB <- as.matrix(Mat_NB)
Mat_SVM <- as.matrix(Mat_SVM)
Mat_Rfa <- as.matrix(Mat_Rfa)

RMicro_NB <- sum(diag(Mat_NB)) / sum(Mat_NB[upper.tri(Mat_NB, diag=TRUE)==TRUE])
PMicro_NB <- sum(diag(Mat_NB)) / sum(Mat_NB[lower.tri(Mat_NB, diag=TRUE)==TRUE])
RMicro_SVM <- sum(diag(Mat_SVM)) / sum(Mat_SVM[upper.tri(Mat_SVM, diag=TRUE)==TRUE])
PMicro_SVM <- sum(diag(Mat_SVM)) / sum(Mat_SVM[lower.tri(Mat_SVM, diag=TRUE)==TRUE])
RMicro_RF <- sum(diag(Mat_Rfa)) / sum(Mat_Rfa[upper.tri(Mat_Rfa, diag=TRUE)==TRUE])
PMicro_RF <- sum(diag(Mat_Rfa)) / sum(Mat_Rfa[lower.tri(Mat_Rfa, diag=TRUE)==TRUE])

2*RMicro_NB*PMicro_NB/(RMicro_NB + PMicro_NB)
2*RMicro_RF*PMicro_RF/(RMicro_RF + PMicro_RF)
2*RMicro_SVM*PMicro_SVM/(RMicro_SVM + PMicro_SVM)

sum(diag(Mat_NB))/sum(Mat_NB)
sum(diag(Mat_SVM))/sum(Mat_SVM)
sum(diag(Mat_Rfa))/sum(Mat_Rfa)

Rfe[c(2,10),c(1,3)] <- 0
round(Rfe, 3)
RMacro_RFe <- mean(Rfe[,2])
PMacro_RFe <- mean(Rfe[,1])
2*RMacro_RFe*PMacro_RFe/(RMacro_RFe + PMacro_RFe)

sum(diag(as.matrix(Mat_Rfe)))/sum(as.matrix(Mat_Rfe))

RMicro_RF <- sum(diag(as.matrix(Mat_Rfe))) / sum(as.matrix(Mat_Rfe[upper.tri(Mat_Rfe, diag=TRUE)==TRUE]))
PMicro_RF <- sum(diag(as.matrix(Mat_Rfe))) / sum(as.matrix(Mat_Rfe[lower.tri(Mat_Rfe, diag=TRUE)==TRUE]))

2*RMicro_RF*PMicro_RF/(RMicro_RF+ PMicro_RF)

#SECTION 3: Topic Models

dtm <- DocumentTermMatrix(Use)
words <- removeSparseTerms(dtm, 0.95)

lda_tm <- LDA(words[1:length(Train),], k = 10)
lda_inf <- posterior(lda_tm, words[(length(Train)+1):length(list),])
head(lda_inf$topics)

g <- lda_inf$topics

colnames(lda_inf$topics) <- apply(terms(lda_tm,10),2,paste,collapse=",")
head(lda_inf$topics)
lda_inf$topics <- apply(lda_inf$topics,1,function(x) colnames(lda_inf$topics)[which.max(x)])
head(lda_inf$topics)

head(round(g[is.element(rownames(g),rownames(Dati[Dati$Topics=="acq",])),],3))
head(round(g[is.element(rownames(g),rownames(Dati[Dati$Topics=="grain",])),],3))
head(round(g[is.element(rownames(g),rownames(Dati[Dati$Topics=="earn",])),],3))
head(round(g[is.element(rownames(g),rownames(Dati[Dati$Topics=="trade",])),],3))