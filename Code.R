# library("caret")
library('pROC')
library('tree')
library('randomForest')
library('class')
library('e1071')	#first install class and then this one
library("ROCR")
library('mclust')
library("ape")
library('cluster')
# library("doParallel")
# library("foreach")

### LOAD DATA ###
###############################################################################
# install.packages("rio")
# importing the required library
library(rio) # Uncomment No.3 line if necessary
# reading data from all sheets
data <- import_list(choose.files())


### CLEANING & ADDING RESPONSE COLUMN ###
###############################################################################
## Drop State Aggregated Data
colSums(is.na(data$county_facts)) 
# 52 NAs in state_abbreviation
to_view <- data$county_facts[is.na(data$county_facts$state_abbreviation)==T,]
# These are the United States row, the rows of the 50 States, and the 
# District of Columbia
# Safe to exclude from model as they are not counties, but aggregated data rows
to_model <- data$county_facts[is.na(data$county_facts$state_abbreviation)==F,]

## Add Response Column
Trump_votes <- data$votes[data$votes$candidate=='Donald Trump',]
Trump_votes$Trump50 = 0
Trump_votes[Trump_votes$fraction_votes > 0.5,'Trump50'] = 1
nrow(Trump_votes) # 3586
nrow(to_model) # 3143
votes_fips <- unique(Trump_votes[,'fips'])
to_model[!to_model$fips %in% votes_fips,'fips'] # 432 FIPS not in votes data
# Specify splitting pattern and save into new column
to_model$not_cty <- vapply(strsplit(to_model$area_name, " County"), 
                           `[`, 1, FUN.VALUE=character(1))
# Alaska 40 House Districts don't match neither FIPS Code nor County Name in
# the 2 datasets. Colorado state has vote data only for the Democrats.
# District of Columbia (Washington, DC) is not included in the votes data.
# Kalawao County of Hawaii is not included in the vote data.
# Congressional Districts (1,2,3, and 4) of Kansas from vote data aggregate
# a plethora of Kansas (KS) counties. We cannot de-aggregate vote data to each
# county as this would be approximate and not depict the reality.
# The State of Maine (ME) has vote data only for the Democrats.
# Minessota (MN) State vote data is not included in the dataset.
# Districts (1-47) of North Dakota from vote data aggregate a plethora of 
# North Dakota (ND) counties. We cannot de-aggregate vote data to each
# county as this would be approximate and not depict the reality.
# Through little search on the internet, ended up to the following useful info
# Middletown is the largest city of Middlesex County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='CT' & 
              Trump_votes$county=='Middletown','county'] = 'Middlesex'
# Cook County is named as Cook Suburbs in the vote data and has different FIPS
Trump_votes[Trump_votes$state_abbreviation=='IL' & 
              Trump_votes$county=='Cook Suburbs','county'] = 'Cook'
# Pittsfield is the largest city of Berkshire County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='MA' & 
              Trump_votes$county=='Pittsfield','county'] = 'Berkshire'
# Taunton is the largest city of Bristol County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='MA' & 
              Trump_votes$county=='Taunton','county'] = 'Bristol'
# Edgartown is the largest city of Dukes County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='MA' & 
              Trump_votes$county=='Edgartown','county'] = 'Dukes'
# Amherst is the largest city of Hampshire County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='MA' & 
              Trump_votes$county=='Amherst','county'] = 'Hampshire'
# Cambridge is the largest city of Middlesex County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='MA' & 
              Trump_votes$county=='Cambridge','county'] = 'Middlesex'
# Boston is the largest city of Suffolk County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='MA' & 
              Trump_votes$county=='Boston','county'] = 'Suffolk'
# Warwick is the largest city of Kent County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='RI' & 
              Trump_votes$county=='Warwick','county'] = 'Kent'
# South Kingstown is the largest city of Washington County
Trump_votes[Trump_votes$state_abbreviation=='RI' & 
              Trump_votes$county=='South Kingstown','county'] = 'Washington'
# St. Johnsbury is the largest city of Caledonia County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='VT' & 
              Trump_votes$county=='St. Johnsbury','county'] = 'Caledonia'
# Morristown is the largest city of Lamoille County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='VT' & 
              Trump_votes$county=='Morristown','county'] = 'Lamoille'
# Derby is the largest city of Orleans County and not the name of it
Trump_votes[Trump_votes$state_abbreviation=='VT' & 
              Trump_votes$county=='Derby','county'] = 'Orleans'
# Bedford city, VA is named as Bedford in the vote dataset
Trump_votes[Trump_votes$state_abbreviation=='VA' & 
              Trump_votes$county=='Bedford','county'] = 'Bedford city'
# Add Response Variable
merged <- merge(x=to_model, y=Trump_votes, 
                by.x=c("state_abbreviation", "not_cty"), 
      by.y=c("state_abbreviation", "county"), all.x=TRUE)
merged <- merged[,c(1:55,62)]
NAs <- merged[is.na(merged$Trump50),] # 512 NAs
more <- merge(NAs, Trump_votes, by.x='fips.x', 
      by.y='fips')
more[,c('fips.x','Trump50.y')]
new_merged <- merge(x=merged, y=more[,c('fips.x','Trump50.y')], 
                    by='fips.x', all.x=TRUE)
new_merged[is.na(new_merged$Trump50),'Trump50'] = 
  new_merged[is.na(new_merged$Trump50),'Trump50.y']
# Wyoming (WY) seems to be a democrat state as in Albany-Natrona,
# Campbell-Johnson, Converse-Niobrara, Crook-Weston, Goshen-Platte,
# Hot Springs-Washakie, Laramie, Sheridan-Big Horn and Uinta-Lincoln Counties
# Trump got 0 votes, while on Sweetwater-Carbon and Fremont-Park Trump got less
# than 50% of the votes for the Republican. Only on Teton-Sublette, Trump got
# more than 50% of the votes for the Republican.
new_merged[new_merged$state_abbreviation=='WY','Trump50'] = 0
new_merged[new_merged$state_abbreviation=='WY' & 
             new_merged$not_cty %in% c('Teton','Sublette'),'Trump50'] = 1
NAs <- new_merged[is.na(new_merged$Trump50),] # 356 NAs
colnames(new_merged)[1] ="fips"
to_model <- new_merged[,c(1,4,2,5:56)]
to_model <- to_model[!is.na(to_model$Trump50),] # 2787 rows = 3143 - 356 (N/As)
rm('merged','more','NAs','new_merged','to_view','Trump_votes')
# https://raw.githubusercontent.com/cphalpert/census-regions/master/
# us%20census%20bureau%20regions%20and%20divisions.csv
region_divisions <- read.table(header = TRUE, sep = ",", text = "
  State,State Code,Region,Division
  Alaska,AK,West,Pacific
  Alabama,AL,South,East South Central
  Arkansas,AR,South,West South Central
  Arizona,AZ,West,Mountain
  California,CA,West,Pacific
  Colorado,CO,West,Mountain
  Connecticut,CT,Northeast,New England
  District of Columbia,DC,South,South Atlantic
  Delaware,DE,South,South Atlantic
  Florida,FL,South,South Atlantic
  Georgia,GA,South,South Atlantic
  Hawaii,HI,West,Pacific
  Iowa,IA,Midwest,West North Central
  Idaho,ID,West,Mountain
  Illinois,IL,Midwest,East North Central
  Indiana,IN,Midwest,East North Central
  Kansas,KS,Midwest,West North Central
  Kentucky,KY,South,East South Central
  Louisiana,LA,South,West South Central
  Massachusetts,MA,Northeast,New England
  Maryland,MD,South,South Atlantic
  Maine,ME,Northeast,New England
  Michigan,MI,Midwest,East North Central
  Minnesota,MN,Midwest,West North Central
  Missouri,MO,Midwest,West North Central
  Mississippi,MS,South,East South Central
  Montana,MT,West,Mountain
  North Carolina,NC,South,South Atlantic
  North Dakota,ND,Midwest,West North Central
  Nebraska,NE,Midwest,West North Central
  New Hampshire,NH,Northeast,New England
  New Jersey,NJ,Northeast,Middle Atlantic
  New Mexico,NM,West,Mountain
  Nevada,NV,West,Mountain
  New York,NY,Northeast,Middle Atlantic
  Ohio,OH,Midwest,East North Central
  Oklahoma,OK,South,West South Central
  Oregon,OR,West,Pacific
  Pennsylvania,PA,Northeast,Middle Atlantic
  Rhode Island,RI,Northeast,New England
  South Carolina,SC,South,South Atlantic
  South Dakota,SD,Midwest,West North Central
  Tennessee,TN,South,East South Central
  Texas,TX,South,West South Central
  Utah,UT,West,Mountain
  Virginia,VA,South,South Atlantic
  Vermont,VT,Northeast,New England
  Washington,WA,West,Pacific
  Wisconsin,WI,Midwest,East North Central
  West Virginia,WV,South,South Atlantic
  Wyoming,WY,West,Mountain
")
to_model <- merge(x=to_model, y=region_divisions, by.x='state_abbreviation', 
                  by.y='State.Code', all.x=TRUE)
to_model <- to_model[,-c(56,57)]
to_model$state_abbreviation <- as.factor(to_model$state_abbreviation)
to_model$Division <- as.factor(to_model$Division)


### EDA ###
###############################################################################
# There not exist any missing values
sum(is.na(to_model))
# No weird data. All data seem to be well recorded
str(to_model)
num_vars <- sapply(to_model,class)=='numeric'
to_model.num <- to_model[,num_vars]
summary(to_model.num[,-1])
par(mfrow=c(4,4))
for (i in 1:length(to_model.num[,-1])){
  qqnorm(to_model.num[,-1][,i], main = paste("Q-Q Plot of" , 
                                             names(to_model.num[,-1])[i]))
  qqline(to_model.num[,-1][,i])
}
p<-ncol(to_model.num[,-1])
par(mfrow=c(2,3))
for (i in 1:p){
  hist(to_model.num[,i], main=names(to_model.num)[i], probability=TRUE)
  lines(density(to_model.num[,i]), col=2)
  index <- seq(min(to_model.num[,i]), max(to_model.num[,i]), length.out=500)
  ynorm <- dnorm(index, mean=mean(to_model.num[,i]), sd(to_model.num[,i]) )
  lines(index, ynorm, col=4, lty=3, lwd=3 )
}
for (i in 1:p){
  boxplot(to_model.num[,i], main = paste("Boxplot of" , names(to_model.num)[i]),
          horizontal=TRUE)
}


### Classification ###
###############################################################################
# Drop variables fips and area_name and put county info in each row
fips <- to_model$fips
to_model <- to_model[,names(to_model)!="fips"]
to_model$county_state <- gsub(" County", "", to_model$area_name)
to_model$county_state <- gsub("[.]","",to_model$county_state)
to_model$county_state <- gsub(" ","_",to_model$county_state)
to_model$county_state <- paste(to_model$county_state, 
                               to_model$state_abbreviation,sep = ".")
rownames(to_model) <- to_model$county_state
to_model[,c("county_state","area_name")] <- NULL

## Check for imbalanced data
round(sum(to_model$Trump50==0)/nrow(to_model),2) # 64%
round(sum(to_model$Trump50==1)/nrow(to_model),2) # 36%
# So, our data are classified in two groups which are represented equally



## Methods
# A bootstrap sample as train set with not included observations as test set
n <- dim(to_model)[1]
ind <- sample(1:n, n, replace=TRUE)
check <- 1:n
t <- check %in% unique(ind)
notin <- check[!t]
train <- to_model[ind,]
train[,'Trump50'] <- as.factor(train[,'Trump50'])
test <- to_model[notin,-53]

# Decision Tree
par(mfrow=c(1,2))
fit1 <- tree(Trump50 ~ .-state_abbreviation, data = train)
fit1_cv <- cv.tree(fit1, FUN = prune.misclass, K=20)
plot(fit1_cv)
plot(fit1_cv$size, fit1_cv$dev / nrow(train), type = "b",
     xlab = "Tree Size", ylab = "CV Misclassification Rate")
fit1_pruned <- prune.misclass(fit1, best = 5)
par(mfrow=c(1,1))
plot(fit1)
text(fit1, pretty = 1)
title(main = "Unpruned Classification Tree")
plot(fit1_pruned)
text(fit1_pruned, pretty = 1)
title(main = "Pruned Classification Tree")

# Random Forest - Variable Importance Plot
fit2 <- randomForest(Trump50 ~ ., data = train, ntree=200, mtry=10,
                     importance=TRUE)
varImpPlot(fit2, main = "Random Forest Variable Importance", n.var = 10)

# SVM
fit3 <- svm(Trump50 ~ ., data=train, scale = TRUE, kernel = "linear")

# ROC Curves
pr_test1 <- predict(fit1_pruned, newdata=test, type='class')
pr_test2.prob <- predict(fit2, newdata=test, type='prob')
pr_test3 <- predict(fit3, newdata=test, type='class')
roc(to_model[notin,'Trump50'], as.numeric(pr_test1)-1, plot=TRUE, grid=TRUE, 
    col="darkgreen", legacy.axes = TRUE, legend = "Tree", asp=NA, ci = TRUE)
par(new=TRUE)
roc(to_model[notin,'Trump50'], pr_test2.prob[,2], plot=TRUE, grid=TRUE, 
    col="red", legacy.axes = TRUE, asp=NA)
par(new=TRUE)
roc(to_model[notin,'Trump50'], as.numeric(pr_test3)-1, plot=TRUE, grid=TRUE, 
    col="blue", legacy.axes = TRUE, asp=NA)
plot_colors <- c("darkgreen","red","blue")
legend(x = "bottomright",inset = 0,
       legend = c("Decision Tree", "Random Forest", "Support Vector Machine"), 
       col=plot_colors, lwd=7, cex=.7, horiz = TRUE)
title(main = "Receiver Operating Characteristic (ROC) Curves")


## 100 Bootstrap samples as training sets and rest as test sets
n <- dim(to_model)[1]
B <- 100
methods <- c('tree', 'forest', 'svm')
accuracy_train <- matrix(data=NA, ncol= B, nrow = length(methods))
accuracy_test <-  matrix(data=NA, ncol= B, nrow = length(methods))
ari <-            matrix(data=NA, ncol= B, nrow = length(methods))
auc <-            matrix(data=NA, ncol= B, nrow = length(methods))
rownames(accuracy_train) <- rownames(accuracy_test) <- rownames(ari) <- 
  rownames(auc) <- methods

for (i in 1:B) {
  print(i)
  set.seed(i)
  ind <- sample(1:n, n, replace=TRUE)
  check <- 1:n
  t <- check %in% unique(ind)
  notin <- check[!t]
  
  train <- to_model[ind,]
  train[,'Trump50'] <- as.factor(train[,'Trump50'])
  test <- to_model[notin,-53]
  # Make sure test subset has all states that train subset has
  states <- 
    setdiff(unique(train$state_abbreviation), unique(test$state_abbreviation))
  while (!identical(states, character(0))){
    cut_paste <- row.names(train[train$state_abbreviation %in% 
                                   states,])[1]
    test <- rbind(test,train[cut_paste,-53])
    train <- train[!(row.names(train) %in% cut_paste),]
    train <- rbind(train,tail(train, n=1))
    notin <- append(notin, which(row.names(to_model) %in% cut_paste))
    states <- 
      setdiff(unique(train$state_abbreviation), unique(test$state_abbreviation))
  }
  # Make sure train subset has all states that test subset has
  states <-
    setdiff(unique(test$state_abbreviation), unique(train$state_abbreviation))
  while (!identical(states, character(0))){
    cut_paste <- row.names(test[test$state_abbreviation %in%
                                  states,])[1]
    train <- rbind(train,to_model[cut_paste,])
    test <- test[!(row.names(test) %in% cut_paste),]
    cut <- tail(row.names(train)[grep("[.]..[.]",row.names(train))],n=1)
    train <- train[!(row.names(train) %in% cut),]
    notin <- notin[!(notin==which(row.names(to_model) %in% cut_paste))]
    states <-
      setdiff(unique(test$state_abbreviation), unique(train$state_abbreviation))
  }
  
  #	tree
  fit1 <- tree(Trump50 ~ .-state_abbreviation, data = train)
  fit1 <- prune.misclass(fit1, best = 5)
  pr_train1 <- predict(fit1, newdata=train, type='class')
  pr_test1 <- predict(fit1, newdata=test, type='class')
  accuracy_train['tree',i] <- sum(train[,'Trump50']==pr_train1)/dim(train)[1]
  accuracy_test['tree',i] <- sum(to_model[notin,'Trump50']==pr_test1)/
    dim(test)[1]
  ari['tree',i] <- adjustedRandIndex(pr_test1, to_model[notin,'Trump50'])
  auc['tree',i] <- auc(roc(to_model[notin,'Trump50'], as.numeric(pr_test1)-1))[1]
  
  # random Forest
  fit2 <- randomForest(Trump50 ~ ., data = train, ntree=200, mtry=10,
                       importance=TRUE)
  pr_train2 <- predict(fit2, newdata=train, type='class')
  pr_test2 <- predict(fit2, newdata=test, type='class')
  accuracy_train['forest',i] <- sum(train[,'Trump50']==pr_train2)/
    dim(train)[1]
  accuracy_test['forest',i] <- sum(to_model[notin,'Trump50']==pr_test2)/
    dim(test)[1]
  ari['forest',i] <- adjustedRandIndex(pr_test2, to_model[notin,'Trump50'])
  auc['forest',i] <- auc(roc(to_model[notin,'Trump50'],
                             as.numeric(pr_test2)-1))[1]
  
  #	svm
  fit3 <- svm(Trump50 ~ ., data=train, scale = TRUE, kernel = "linear")
  pr_train3 <- predict(fit3, newdata=train, type='class')
  pr_test3 <- predict(fit3, newdata=test, type='class')
  accuracy_train['svm',i] <- sum(train[,'Trump50']==pr_train3)/dim(train)[1]
  accuracy_test['svm',i] <- sum(to_model[notin,'Trump50']==pr_test3)/dim(test)[1]
  ari['svm',i] <- adjustedRandIndex(pr_test3, to_model[notin,'Trump50'])
  auc['svm',i] <- auc(roc(to_model[notin,'Trump50'], as.numeric(pr_test3)-1))
}

# Cumulative plots
summary <- data.frame(accuracy_train=round(rowMeans(accuracy_train),2),
                      accuracy_test=round(rowMeans(accuracy_test),2),
                      ARI=round(rowMeans(ari),2),AUC=round(rowMeans(auc),2))
# > summary
#           accuracy_train  accuracy_test   ARI         AUC
# tree      0.8025475       0.7686658       0.2779456   0.7253239
# forest    1.0000000       0.8854355       0.5907535   0.8666254
# svm       0.8854037       0.8475608       0.4751564   0.8082475
boxplot(t(accuracy_test), ylab='Predictive Accuracy', xlab='method')
title(main = "Predictive Accuracy per Trained Model")
boxplot(t(ari), ylab='Adjusted Rand Index', xlab='method')
title(main = "Adjusted Rand Index per Trained Model")
boxplot(t(auc), ylab='Area Under the Curve', xlab='method')
title(main = "Area Under the Curve per Trained Model")


### Clustering ###
###############################################################################
## Dataset with the variables to be clustered
to_cluster <- to_model[,c(2:23)]
# Scale the data
scaled_data <- as.data.frame(scale(to_cluster))
scaled_data <- scaled_data[,(names(scaled_data)!="RHI525214")]
scaled_data <- scaled_data[,(names(scaled_data)!="RHI325214")]
scaled_data <- scaled_data[,(names(scaled_data)!="RHI625214")]
# rowSums(cor(scaled_data)>=0.7)
# max(rowMeans(cor(scaled_data)))
# scaled_data <- scaled_data[,(names(scaled_data)!="PST045214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="VET605213")]
# scaled_data <- scaled_data[,(names(scaled_data)!="POP815213")]
# # scaled_data <- scaled_data[,(names(scaled_data)!="RHI425214")]
#     
# 
# model <- lm(Trump50 ~ .-RHI125214-RHI225214-VET605213-AGE295214-POP645213, data = scaled_data)
# library(car)
# #calculate the VIF for each predictor variable in the model
# vif(model)
# library(corrplot)
# corrplot(scaled_data,method='number',is.corr = F)
# scaled_data[,'Trump50'] <- as.factor(scaled_data[,'Trump50'])



## Hierarchical clustering - Find best linkage and distance method
# Define nodePar
nodePar <- list(lab.cex = 0.6, pch = c(NA, 19), 
                cex = 0.7, col = "blue")
# edgePar <- list(col = 1:3, lwd = 2:1)
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="complete")
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="complete")
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="complete")
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="ward.D") # 3
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="ward.D") # 2
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="ward.D") # 5
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="ward.D2") # 3
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="ward.D2") # 4
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="ward.D2") # 6
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="single")
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="single")
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="single")
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="average")
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="average")
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="average")
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="median")
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="median")
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="median")
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="centroid")
hc1 <- hclust(dist(scaled_data, method = "maximum"), method="centroid")
hc1 <- hclust(dist(scaled_data, method = "manhattan"), method="centroid")

# distances:  euclidean,maximum,manhattan
# linkages:   ward.D,ward.D2,single,complete,average,mcquitty,median,centroid
# Convert hclust into a dendrogram and plot
hc1 <- hclust(dist(scaled_data, method = "euclidean"), method="ward.D")
hcd <- as.dendrogram(hc1)
par(mfrow=c(1,2))
plot(hcd, nodePar = nodePar, leaflab = "none")
plot(as.phylo(hc1), type = "unrooted", cex = 0.5, show.tip.label = FALSE)
mtext(expression(bold("Ward Linkage & Euclidean Distance")), side = 3, 
      line = - 2, outer = TRUE, cex = 2)

hc2 <- hclust(dist(scaled_data, method = "maximum"), method="ward.D")
hcd <- as.dendrogram(hc2)
par(mfrow=c(1,2))
plot(hcd, nodePar = nodePar, leaflab = "none")
plot(as.phylo(hc2), type = "unrooted", cex = 0.5, show.tip.label = FALSE)
mtext(expression(bold("Ward Linkage & Maximum Distance")), side = 3, 
      line = - 2, outer = TRUE, cex = 2)

hc3 <- hclust(dist(scaled_data, method = "manhattan"), method="ward.D") # 5
hcd <- as.dendrogram(hc3)
par(mfrow=c(1,2))
plot(hcd, nodePar = nodePar, leaflab = "none")
plot(as.phylo(hc3), type = "unrooted", cex = 0.5, show.tip.label = FALSE)
mtext(expression(bold("Ward Linkage & Manhattan Distance")), side = 3, 
      line = - 2, outer = TRUE, cex = 2)


rect.hclust(hc1, h=31, border="red")
rect.hclust(hc1, h=20, border="red")


## Hierarchical clustering - Find best number of clusters
par(mfrow=c(1,1))
plot(silhouette(cutree(hc1, k = 2), dist(scaled_data, method = "euclidean")),col=2:3,main ='Euclidean', border=NA) # 0.3
plot(silhouette(cutree(hc1, k = 3), dist(scaled_data, method = "euclidean")),col=2:4,main ='Euclidean', border=NA) 
plot(silhouette(cutree(hc1, k = 4), dist(scaled_data, method = "euclidean")),col=2:5,main ='Euclidean', border=NA) 
plot(silhouette(cutree(hc1, k = 5), dist(scaled_data, method = "euclidean")),col=2:6,main ='Euclidean', border=NA) 
plot(silhouette(cutree(hc1, k = 6), dist(scaled_data, method = "euclidean")),col=2:7,main ='Euclidean', border=NA)
plot(silhouette(cutree(hc2, k = 2), dist(scaled_data, method = "maximum")),col=2:3,main ='Maximum', border=NA) # 0.39
plot(silhouette(cutree(hc2, k = 3), dist(scaled_data, method = "maximum")),col=2:4,main ='Maximum', border=NA) 
plot(silhouette(cutree(hc2, k = 4), dist(scaled_data, method = "maximum")),col=2:5,main ='Maximum', border=NA)
plot(silhouette(cutree(hc2, k = 5), dist(scaled_data, method = "maximum")),col=2:6,main ='Maximum', border=NA) 
plot(silhouette(cutree(hc2, k = 6), dist(scaled_data, method = "maximum")),col=2:7,main ='Maximum', border=NA) 
plot(silhouette(cutree(hc3, k = 2), dist(scaled_data, method = "manhattan")),col=2:3,main ='Manhattan', border=NA) # 0.45
plot(silhouette(cutree(hc3, k = 3), dist(scaled_data, method = "manhattan")),col=2:4,main ='Manhattan', border=NA) 
plot(silhouette(cutree(hc3, k = 4), dist(scaled_data, method = "manhattan")),col=2:5,main ='Manhattan', border=NA) # 0.29
plot(silhouette(cutree(hc3, k = 5), dist(scaled_data, method = "manhattan")),col=2:6,main ='Manhattan', border=NA) 
plot(silhouette(cutree(hc3, k = 6), dist(scaled_data, method = "manhattan")),col=2:7,main ='Manhattan', border=NA) 

scaled_data$y <- cutree(hc3, k = 2)
scaled_data[,'y'] <- as.factor(scaled_data[,'y'])


# Find Variable Importance 
fit2 <- randomForest(y ~ ., data = scaled_data, ntree=500, mtry=floor(ncol(scaled_data)/2), importance=TRUE)
varImpPlot(fit2, main = "Random Forest Variable Importance")
# important <- c("POP010210","AGE295214","AGE775214","RHI125214","RHI225214",
#                "RHI325214","RHI625214","RHI725214","RHI825214","POP715213",
#                "POP815213","EDU635213","VET605213")
# important <- c("RHI325214","AGE295214","RHI225214","RHI725214","EDU635213",
#                "POP715213","RHI625214","VET605213","RHI825214","POP010210")
# not_important <- c("RHI425214", "RHI525214", "SEX255214")
# scaled_data <- scaled_data[,!(names(scaled_data) %in% not_important)]


scaled_data <- scaled_data[,(names(scaled_data)!="y")]
# scaled_data <- scaled_data[,(names(scaled_data)!="POP010210")]
# scaled_data <- scaled_data[,(names(scaled_data)!="PST040210")]
# scaled_data <- scaled_data[,(names(scaled_data)!="POP715213")]
# scaled_data <- scaled_data[,(names(scaled_data)!="EDU685213")]
# scaled_data <- scaled_data[,(names(scaled_data)!="EDU635213")]
# scaled_data <- scaled_data[,(names(scaled_data)!="RHI125214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="RHI725214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="PST120214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="SEX255214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="RHI825214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="VET605213")]
# scaled_data <- scaled_data[,(names(scaled_data)!="PST045214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="AGE295214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="RHI225214")]
# scaled_data <- scaled_data[,(names(scaled_data)!="AGE135214")]


## Explain clusters found based on economic related
to_explain <- to_model[,c(24:52)]
add_clusters <- scaled_data[,c("y","PST045214")]
to_explain <- merge(to_explain, add_clusters, by = 'row.names', all = TRUE)
rownames(to_explain) <- to_explain$Row.names
to_explain <- to_explain[,!(names(to_explain) %in% c("Row.names","PST045214"))]

explainings <- aggregate(. ~ y, data = to_explain, FUN = mean)
# data$dictionary$column_name
# names(explainings)
new <- explainings[,c(-1,-2,-8,-11,-18,-20,-29)]
par(mfrow=c(4,2))
for (i in 1:ncol(new)){
  barplot(height=new[,i], names=explainings$y, col=explainings$y,horiz=T, las=1,
          main=names(new)[i])
}

