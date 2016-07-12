# Online News Popularity
# Data Mining - Final Project Code - R Script
# Ishan Gupta, Rohit Sharma, Saurabh Agrawal, Varun Jindal
#------------------------------------------------------

library(class)
library(stats)
library(caret)
library(randomForest)
library(ROCR)
library(pROC)
library(rattle)
library(RColorBrewer)
library(ggplot2)

library(e1071)
library(class)
library(clusterSim)
library(rpart) #R part
library(rpart.plot)
library(C50)

news<-read.csv("OnlineNewsPopularity.csv")
View (head(news))

summary(news)


#Data Preprocessing
#remove url and timedelta from the dataset using subset
newsSubset <- subset( news, select = -c(url, timedelta ) )

#remove outliers from dataset
newsSubset=newsSubset[!newsSubset$n_unique_tokens==701,]

# generate z-scores using the scale() function
for(i in ncol(newsSubset)-1){ 
  newsSubset[,i]<-scale(newsSubset[,i], center = TRUE, scale = TRUE)
}
#sample data before classifiying dataset into training and test
set.seed(100)
# Dataset for classification
newsdataset <-newsSubset[1:15000,]
#set shares to 1 if greater than 1400 else 0 
newsdataset$shares <- as.factor(ifelse(newsdataset$shares >1400,1,0))

newstrain <- newsdataset[1:10000,]
newstest <- newsdataset[10001:15000,]

#colorPalette
color.lr<-'#adef69'
color.knn<-'#ab69ef'
color.cart<-'#ef696a'
color.c50<-'#efab69'
color.rf<-'#69adef'


#KNN BEGIN
newsknn10<-knn(train=newstrain,test=newstest,cl=newstrain$shares,k=10)
newsknn10class <- predict( newsknn3,newstest,type="class")
#ConfusionMatrix for knn10
confusionMatrix(newsknn10, newstest$shares)

newsknn5<-knn(train=newstrain,test=newstest,cl=newstrain$shares,k=5)
newsknn5class <- predict( newsknn5,newstest,type="class")
#confusionMatrix for knn5
confusionMatrix(newsknn5, newstest$shares)

newsknn1<-knn(train=newstrain,test=newstest,cl=newstrain$shares,k=1)
newsknn1class <- predict( newsknn1,newstest,type="class")
#confusionMatrix for knn1
confusionMatrix(newsknn1class, newstest$shares)



newsknn3 <- knn3(shares ~.,newstrain)
newsknn3class <- predict( newsknn3,newstest,type="class")
#confusionMatrix for knn3
newsknn3prob <- predict( newsknn3,newstest,type="prob")
# Confusion matrix
confusionMatrix(newsknn3class, newstest$shares)

# ROC Curve KNN
newsKnnRoc <- roc(newstest$shares,newsknn3prob[,2])
plot(newsKnnRoc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.knn, print.thres=TRUE)

#KNN END


newsCart<-rpart(shares ~.,newstrain,method='class')
fancyRpartPlot(newsCart)
newsCartClassPrediction<-predict( newsCart,newstest ,type="class")
newsCartProbPrediction<-predict( newsCart,newstest ,type="prob")
# Confusion matrix
confusionMatrix(newsCartClassPrediction, newstest$shares)

# ROC Curve
newsCartRoc <- roc(newstest$shares,newsCartProbPrediction[,2])
plot(newsCartRoc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.cart, print.thres=TRUE)
#CART END


#C50 Begin
#number of trials 5
newsC50<-C5.0(shares ~.,newstrain,trials=5)
#Summarize summary here
summary(newsC50)
newsC50classPrediction<-predict( newsC50,newstest,type="class" )
#confusionMatrix for trial=5
confusionMatrix(newsC50classPrediction, newstest$shares)

#number of trials 7
newsC50<-C5.0(shares ~.,newstrain,trials=7)
#Summarize summary here
summary(newsC50)
newsC50classPrediction<-predict( newsC50,newstest,type="class" )
#confusionMatrix for trial=7
confusionMatrix(newsC50classPrediction, newstest$shares)

#trial 9 further increase in trial results in lesser accuracy
newsC50<-C5.0(shares ~.,newstrain,trials=9)
summary(newsC50)
#predict
newsC50classPrediction<-predict( newsC50,newstest,type="class" )
newsC50probPrediction<-predict( newsC50,newstest,type="prob" )
# Confusion matrix for trial=9
confusionMatrix(newsC50classPrediction, newstest$shares)


# ROC Curve
newsC50Roc <- roc(newstest$shares,newsC50probPrediction[,2])
plot(newsC50Roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.c50, print.thres=TRUE)

#C50 END

#Random Forest Begin
#random forest nodesize=20 ntree=70
newsrf20<-randomForest(shares ~.,newstrain,ntree=70,nodesize=20,nPerm=10,
                       mtry=8,proximity=TRUE,importance=TRUE)

#random forest nodesize=10 ntree=80
newsrf10<-randomForest(shares ~.,newstrain,ntree=80,nodesize=10,nPerm=10,
                       mtry=8,proximity=TRUE,importance=TRUE)
#randomForest nodesize=10 ntree=100
newsrf15<-randomForest(shares ~.,newstrain,ntree=100,nodesize=15,nPerm=10,
                        mtry=8,proximity=TRUE,importance=TRUE)

#combining all randomForest trained datasets
rf.all<-combine(newsrf10,newsrf20,newsrf15)


newspredallclass<-predict( rf.all,newstest, type="class")
newspredallprob<-predict( rf.all,newstest, type="prob")

confusionMatrix(newspredallclass, newstest$shares)
# ROC Curve
newsRfRoc <- roc(newstest$shares,newspredallprob[,2])
plot(newsRfRoc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.rf, print.thres=TRUE)

#Random Forest End

#Comparing All models

ROCCurve<-par(pty = "s")
plot(performance(prediction(newsknn3prob[,2],newstest$shares),'tpr','fpr'),
     col=color.knn, lwd=3
)
text(0.2,0.8,"KNN3",col=color.knn)
plot(performance(prediction(newsCartProbPrediction[,2],newstest$shares),'tpr','fpr'),
     col=color.cart, lwd=3, add=TRUE
)
text(0.45,0.32,"CART",col=color.cart)
plot(performance(prediction(newsC50probPrediction[,2],newstest$shares),'tpr','fpr'),
     col=color.c50, lwd=3, add=TRUE
)
text(0.10,0.6,"C5.0",col=color.c50)
plot(performance(prediction(newspredallprob[,2],newstest$shares),'tpr','fpr'),
     col=color.rf, lwd=3, add=TRUE
)
text(0.5,0.1,"RF- Combined",col=color.rf)