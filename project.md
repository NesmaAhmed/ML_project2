Prediction Assignment Writeup
-----------------------------

introduction
============

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways.

library
=======

    library(caret)
    library(rpart)
    library(rpart.plot)
    library(randomForest)
    library(corrplot)
    library(rattle)
    library(knitr)
    library(gbm)
    library(markdown)

LOAD DATA
=========

    read_train=read.csv("F:/data science specialist/course8/pml-training.csv")
    read_test=read.csv("F:/data science specialist/course8/pml-testing.csv")

found the diagnoses predictors that have one unique value (i.e. are zero
variance predictors)

    NZV <- nearZeroVar(read_train)
    read_train <- read_train[, -NZV]

Remove variables with missing values

    var_with_NA   <- sapply(read_train, function(x) mean(is.na(x))) > 0.95
    read_train <- read_train[, var_with_NA==FALSE]
    read_train <- read_train[, -(1:5)]

partition the training data into two part (70% train , 30% test to test
the model)

    inTrain  <- createDataPartition(read_train$classe, p=0.7, list=FALSE)
    Train_data <- read_train[inTrain, ]
    Test_data <- read_train[-inTrain, ]
     #dim(Train_data);  dim(Test_data)

buliding prediction model in different technique
================================================

1 Random Forest Model
=====================

    set.seed(12345)
    controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
    modFitRandForest <- train(classe ~ ., data=Train_data, method="rf",
                           trControl=controlRF)

using test data
===============

    predictRandForest <- predict(modFitRandForest, newdata=Test_data)
    confMatRandForest <- confusionMatrix(predictRandForest, Test_data$classe)
    confMatRandForest$overall['Accuracy']

    ##  Accuracy 
    ## 0.9983008

Ploting Matrix Results to Random Forest Model
=============================================

    #plot(confMatRandForest$table, col = "beige",
     #main = paste("Random Forest - Accuracy =",
     #              round(confMatRandForest$overall['Accuracy'], 4)))

2 Decision Trees Method
=======================

    modFitDecTree <- rpart(classe ~ ., data=Train_data, method="class")
    suppressWarnings(fancyRpartPlot(modFitDecTree))

![](project_files/figure-markdown_strict/unnamed-chunk-9-1.png)

    predictDecTree <- predict(modFitDecTree, newdata=Test_data, type="class")
    confMatDecTree <- confusionMatrix(predictDecTree, Test_data$classe)
    confMatDecTree$overall['Accuracy']

    ##  Accuracy 
    ## 0.7259133

3 Generalized Boosted Model
===========================

    controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
    modFitGBM  <- train(classe ~ ., data=Train_data, method = "gbm",
                        trControl = controlGBM, verbose = FALSE)
    predictGBM <- predict(modFitGBM, newdata=Test_data)
    confMatGBM <- confusionMatrix(predictGBM, Test_data$classe)
    confMatGBM$overall['Accuracy']

    ##  Accuracy 
    ## 0.9870858

4 Cross Validation Model
========================

    cv3 = trainControl(method="cv",number=3,allowParallel=TRUE,verboseIter=TRUE)
    modrf = train(classe~., data=Train_data, method="rf",trControl=cv3)

    ## + Fold1: mtry= 2 
    ## - Fold1: mtry= 2 
    ## + Fold1: mtry=27 
    ## - Fold1: mtry=27 
    ## + Fold1: mtry=53 
    ## - Fold1: mtry=53 
    ## + Fold2: mtry= 2 
    ## - Fold2: mtry= 2 
    ## + Fold2: mtry=27 
    ## - Fold2: mtry=27 
    ## + Fold2: mtry=53 
    ## - Fold2: mtry=53 
    ## + Fold3: mtry= 2 
    ## - Fold3: mtry= 2 
    ## + Fold3: mtry=27 
    ## - Fold3: mtry=27 
    ## + Fold3: mtry=53 
    ## - Fold3: mtry=53 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting mtry = 27 on full training set

    modtree = train(classe~.,data=Train_data,method="rpart",trControl=cv3)

    ## + Fold1: cp=0.03967 
    ## - Fold1: cp=0.03967 
    ## + Fold2: cp=0.03967 
    ## - Fold2: cp=0.03967 
    ## + Fold3: cp=0.03967 
    ## - Fold3: cp=0.03967 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting cp = 0.0397 on full training set

    prf=predict(modrf,Test_data)
    ptree=predict(modtree,Test_data)
    table(prf,ptree)

    ##    ptree
    ## prf    A    B    C    D    E
    ##   A 1490   30  148    0    6
    ##   B  485  299  358    0    0
    ##   C  497   16  512    0    0
    ##   D  446  165  350    0    0
    ##   E  171   60  373    0  479

    confMatcv<- confusionMatrix(prf, Test_data$classe)
    confMatcv$overall['Accuracy']

    ##  Accuracy 
    ## 0.9986406

conclusion

so my opinion that use the cross validation random forest model

    answers=predict(modrf,read_test)
    answers

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
