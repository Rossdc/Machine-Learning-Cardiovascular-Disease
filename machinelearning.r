getwd()
cardio <- read.csv("cardio_train.csv", header = T, sep = ";")

summary(cardio)
str(cardio)

library(dplyr)

#Pre-processing

cardio[!complete.cases(cardio),]
cardio <- select(cardio, -c(id))

#Subsetting data

cardiopruned <- subset(cardio, select = c("cardio", "active", "cholesterol", "smoke", "alco", "gender"))

#Making variables factors as required

cardiopruned$cardio = as.factor(cardiopruned$cardio)
cardiopruned$gender = as.factor(cardiopruned$gender)

#Creating test and training data

dt = sort(sample(nrow(cardiopruned), nrow(cardiopruned)*.7))
train<-cardiopruned[dt,]
val<-cardiopruned[-dt,]
nrow(train)

library(rpart)

#Creating decision tree

tree1 <- rpart(cardio~., data = train, method = "class", control = rpart.control(cp =0))

summary(tree1)

plot(tree1)
text(tree1)

#Beautifying Initial Decision Tree

library(rattle)
library(rpart.plot)
library(RColorBrewer)

prp(tree1, faclen = 0, cex = 0.8, extra = 1)

tot_count <- function(x, labs, digits, varlen)
{paste(labs, "\n\nn =", x$frame$n)}
prp(tree1, faclen = 0, cex = 0.8, node.fun=tot_count)

fancyRpartPlot(tree1)

#####DECISION TREE#####

plotcp(tree1)
printcp(tree1)

#Selecting best cp

bestcp <- tree1$cptable[which.min(tree1$cptable[,"xerror"]),"CP"]

pruned <- prune(tree1, cp = bestcp)

#Decision tree with best cp

prp(pruned, faclen = 0, cex = 0.8, extra = 1)

#Beautifying decision tree with best cp

fancyRpartPlot(pruned, uniform=TRUE)

#Creating confusion matrix to enable evaluation

conf.matrix <- table(train$cardio, predict(pruned,type="class"))
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Predicted", colnames(conf.matrix), sep = ":")
print(conf.matrix)

library(caret)

#Confusion matrix on training data

confusionMatrix(predict(pruned, type = "class"), train$cardio)

#Confusion matrix on testing data

confusionMatrix(predict(pruned,val,type = "class"), val$cardio)

library(ROCR)

#Storing Model Performance Scores
val1 = predict(pruned, val, type = "prob")

pred_val <-prediction(val1[,2],val$cardio)

# Calculating Area under Curve
perf_val <- performance(pred_val,"auc")

plot(performance(pred_val, measure="lift", x.measure="rpp"), colorize=TRUE)

# Calculating True Positive and False Positive Rate
perf_val <- performance(pred_val, "tpr", "fpr")

# Plot the ROC curve
plot(perf_val, col = "red", lwd = 1.5)

#Getting KS statistic

ks1.tree <- max(attr(perf_val, "y.values")[[1]] - (attr(perf_val, "x.values")[[1]]))
ks1.tree

#####Random Forest#####

library(randomForest)
 
forest_model <- randomForest(cardio~., data = cardiopruned, subset = dt)

#Getting random forest results

pred.rf <- predict(forest_model, newdata = val, type = "class")

table(val$cardio, pred.rf)

#Creating confusion matrix for random forest results

confusionMatrix(pred.rf, val$cardio)