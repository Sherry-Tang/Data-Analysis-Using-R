# HW 3 GROUP PART XIAOWEN TANG
#install.packages('gplots')
#install.packages("ROCR")
#install.packages('arules')
library(arules)
library(gplots)
library(ROCR)
library(dplyr)
library(ggplot2)
library('e1071') #Naive Bayes
library('randomForest') #RandomForest
library(caret)
library(rpart)
library(rpart.plot)
library(class)
library(ggstatsplot)
library(pROC)
library(mltools)
require(tidyr)
require(dplyr)
library(xgboost)
require(Matrix)
#Load data
setwd("C:/Users/sherr/Sherry/Carlson/Courses/6130 Introduction to BA in R-Mochen Yang/HW3")
xyz<-read.csv('XYZData.csv')
xyz<-xyz[,c(2:27)]
xyz<-xyz%>%mutate(AgeCat=as.factor(ifelse(age < 12, "kid", 
                                          ifelse(age < 18, "teen",
                                                 ifelse(age < 30, "adult",
                                                        ifelse(age < 57, "middle",
                                                               ifelse(age < 80, "elder",'None')))))))
#Normalize the data
normalize = function(x){
  return ((x - min(x))/(max(x) - min(x)))}
xyz_norm = xyz %>% mutate_at(1:25, normalize)
#Remove ourliers # this step helps to increase AUC 10%
outliers <-boxplot(xyz_norm[,c(8:12)])$out
x<-xyz_norm
x<- x[-which(x$songsListened %in% outliers|
               x$lovedTracks %in% outliers|
               x$posts %in% outliers|
               x$playlists %in% outliers|
               x$shouts %in% outliers),]
outliers <-boxplot(x[,c(8:12)])$out
# One hot encoding the gender column
x<-x%>%mutate(male=as.factor(male),
              good_country=as.factor(good_country))
x<-x[,-1]
#Split train test data
train_rows = createDataPartition(y = x$adopter, p = 0.85, list = FALSE)
xyz_train = x[train_rows,]
xyz_test = x[-train_rows,]
# 1. XGBoost 
sparse_matrix <- sparse.model.matrix(adopter ~lovedTracks + male + subscriber_friend_cnt + 
                                       good_country + delta_songsListened + avg_friend_age + delta_playlists + 
                                       AgeCat + playlists + tenure + songsListened + delta_lovedTracks + 
                                       avg_friend_male + delta_good_country + shouts, data = xyz_train)
output_vector = xyz_train$adopter
bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 1,
               eta = 1, nthread = 3, nrounds = 15,objective = "binary:logistic")
# Prediction
matrix_test <- sparse.model.matrix(adopter ~ lovedTracks + male + subscriber_friend_cnt + 
                                     good_country + delta_songsListened + avg_friend_age + delta_playlists + 
                                     AgeCat + playlists + tenure + songsListened + delta_lovedTracks + 
                                     avg_friend_male + delta_good_country + shouts, data = xyz_test)
p0=predict(bst,matrix_test,type='raw')
#Calculate AUC for this model
roc_obj <- roc(xyz_test$adopter, p0)
print(auc(roc_obj))
# Add precision, recall and f measure for the data
xyz_test_xgb<-xyz_test %>%
  mutate(probility=p0)%>%
  arrange(desc(probility))%>%
  mutate(precision_pos=cumsum(adopter)/(cumsum(adopter)+cumsum(1-adopter)),
         recall_pos=cumsum(adopter)/sum(adopter),
         percentage_data=row_number()/nrow(xyz_test),
         tpr=cumsum(adopter)/sum(adopter),
         fpr=cumsum(1-adopter)/sum(1-adopter))%>%
  mutate(f_measure=2*precision_pos*recall_pos/(precision_pos+recall_pos))
# Choose best threshold if use f-measure
a=xyz_test_xgb%>%filter(f_measure==max(xyz_test_xgb$f_measure))
print(a$f_measure)
xyz_predicted<-xyz_test_xgb%>%mutate(predic_values=if_else(probility>=a$probility,"predict 1","predict 0"),
                                   actual_values=if_else(adopter==1,'actual 1','actual 0'))
table(xyz_predicted$predic_values, xyz_predicted$actual_values,dnn=c("Prediction","Actual"))


#2. Naive Bayes
fit1 = naiveBayes(as.factor(adopter) ~., data=xyz_train,na.action = na.pass)
# Prediction
p1=predict(fit1,xyz_test,type = "raw")[,2]
#Calculate AUC for this model
roc_obj <- roc(xyz_test$adopter, p1)
print(auc(roc_obj))
# Add precision, recall and f measure for the data
xyz_test_nb<-xyz_test %>%
  mutate(probility=p1)%>%
  arrange(desc(probility))%>%
  mutate(precision_pos=cumsum(adopter)/(cumsum(adopter)+cumsum(1-adopter)),
         recall_pos=cumsum(adopter)/sum(adopter),
         percentage_data=row_number()/nrow(xyz_test),
         tpr=cumsum(adopter)/sum(adopter),
         fpr=cumsum(1-adopter)/sum(1-adopter))%>%
  mutate(f_measure=2*precision_pos*recall_pos/(precision_pos+recall_pos))%>%
  replace(is.na(.), 0)
# Choose best threshold if use f-measure
a=xyz_test_nb%>%filter(f_measure==max(xyz_test_nb$f_measure))
print(a$f_measure)
xyz_predicted<-xyz_test_nb%>%mutate(predic_values=if_else(probility>=a$probility,"predict 1","predict 0"),
                                     actual_values=if_else(adopter==1,'actual 1','actual 0'))
table(xyz_predicted$predic_values, xyz_predicted$actual_values,dnn=c("Prediction","Actual"))


#3. Logistic Regression
# Model construction
#Stepwise model selection to find best model
full=glm(adopter ~.,family='binomial',data=xyz_train)
null=glm(adopter ~1,family='binomial',data=xyz_train)
step(null,scope=list(lower=null,upper=full),direction="both")
#best model based on AIC round 1
fit2<-glm(formula = adopter ~ lovedTracks + male + subscriber_friend_cnt + 
            good_country + delta_songsListened + avg_friend_age + delta_playlists + 
            AgeCat + playlists + tenure + songsListened + delta_lovedTracks + 
            avg_friend_male + delta_good_country + shouts,  
          family = "binomial"(link='logit'), data = xyz_train)
summary(fit2)
#Drop unsignificant column :delta_good_country 
#test for Collinearity
library(car)
vif(fit2)
#None greater than 10, so there is no collinearity.
# Prediction
p2<-predict(fit2,newdata=xyz_test,type='response')
#Calculate AUC for this model
roc_obj <- roc(xyz_test$adopter, p2)
print(auc(roc_obj))
# Add precision, recall and f measure for the data
xyz_test_lg<-xyz_test %>%
  mutate(probility=p2)%>%
  arrange(desc(probility))%>%
  mutate(precision_pos=cumsum(adopter)/(cumsum(adopter)+cumsum(1-adopter)),
         recall_pos=cumsum(adopter)/sum(adopter),
         percentage_data=row_number()/nrow(xyz_test),
         tpr=cumsum(adopter)/sum(adopter),
         fpr=cumsum(1-adopter)/sum(1-adopter))%>%
  mutate(f_measure=2*precision_pos*recall_pos/(precision_pos+recall_pos))%>%
  replace(is.na(.), 0)
# Choose best threshold if use f-measure
a=xyz_test_lg%>%filter(f_measure==max(xyz_test_lg$f_measure))
print(a$f_measure)
xyz_predicted<-xyz_test_lg%>%mutate(predic_values=if_else(probility>=a$probility,"predict 1","predict 0"),
                                    actual_values=if_else(adopter==1,'actual 1','actual 0'))
table(xyz_predicted$predic_values, xyz_predicted$actual_values,dnn=c("Prediction","Actual"))

#Random Forest
xyz.rf<-randomForest(adopter ~.,
                 xyz_train,ntree=500,importance=T)
summary(rf)
plot(rf)
round(importance(rf), 2)
# Prediction
p3<-predict(xyz.rf, xyz_test, type='response')
#Calculate AUC for this model
roc_obj <- roc(xyz_test$adopter, p3)
print(auc(roc_obj))
# Add precision, recall and f measure for the data
xyz_test_rf<-xyz_test %>%
  mutate(probility=p3)%>%
  arrange(desc(probility))%>%
  mutate(precision_pos=cumsum(adopter)/(cumsum(adopter)+cumsum(1-adopter)),
         recall_pos=cumsum(adopter)/sum(adopter),
         percentage_data=row_number()/nrow(xyz_test),
         tpr=cumsum(adopter)/sum(adopter),
         fpr=cumsum(1-adopter)/sum(1-adopter))%>%
  mutate(f_measure=2*precision_pos*recall_pos/(precision_pos+recall_pos))%>%
  replace(is.na(.), 0)
# Choose best threshold if use f-measure
a=xyz_test_rf%>%filter(f_measure==max(xyz_test_rf$f_measure))
print(a$f_measure)
xyz_predicted<-xyz_test_rf%>%mutate(predic_values=if_else(probility>=a$probility,"predict 1","predict 0"),
                                    actual_values=if_else(adopter==1,'actual 1','actual 0'))
table(xyz_predicted$predic_values, xyz_predicted$actual_values,dnn=c("Prediction","Actual"))

#Plot the three model roc
ggplot()+
  geom_line(aes(x=xyz_test_xgb$fpr,y=xyz_test_nb$tpr),color='red')+
  geom_line(aes(x=xyz_test_nb$fpr,y=xyz_test_xgb$tpr),color='blue')+
  geom_line(aes(x=xyz_test_lg$fpr,y=xyz_test_lg$tpr),color='gold')+
  geom_line(aes(x=xyz_test_rf$fpr,y=xyz_test_rf$tpr),color='green')+
  theme(legend.position = c(0.95, 0.95))+
labs(x= 'False Positive Rate', y= 'True Positive Rate',color= 'State')

