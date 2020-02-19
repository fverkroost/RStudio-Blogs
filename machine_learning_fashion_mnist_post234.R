################################################################################################
######################################## DATA PREPARATION ######################################
################################################################################################

# Load data
library(devtools)
devtools::install_github("rstudio/keras")
library(keras)        
install_keras()  
fashion_mnist = keras::dataset_fashion_mnist()

# Prepare data
library(magrittr)
c(train.images, train.labels) %<-% fashion_mnist$train
c(test.images, test.labels) %<-% fashion_mnist$test
train.images = data.frame(t(apply(train.images, 1, c))) / max(fashion_mnist$train$x)
test.images = data.frame(t(apply(test.images, 1, c))) / max(fashion_mnist$train$x)

# Combine training and test images and labels
pixs = ncol(fashion_mnist$train$x)
names(train.images) = names(test.images) = paste0('pixel', 1:(pixs^2))
train.labels = data.frame(label = factor(train.labels))
test.labels = data.frame(label = factor(test.labels))
train.data = cbind(train.labels, train.images)
test.data = cbind(test.labels, test.images)

# Separate factor vectors with outcomes
cloth_cats = c('Top', 'Trouser', 'Pullover', 'Dress', 'Coat',  
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')
train.classes = factor(cloth_cats[as.numeric(as.character(train.labels$label)) + 1])
test.classes = factor(cloth_cats[as.numeric(as.character(test.labels$label)) + 1])

################################################################################################
############################################## PCA #############################################
################################################################################################

# Average pixel values over all images in training data set
train.images.ave = data.frame(pixel = apply(train.images, 2, mean), 
                              x = rep(1:pixs, each = pixs), 
                              y = rep(1:pixs, pixs))

# Plot average pixel values with custom plotting aesthetics
library(ggplot2)
my_theme = function () { 
  theme_bw() + 
    theme(axis.text = element_text(size = 14),
          axis.title = element_text(size = 14),
          strip.text = element_text(size = 14),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = "bottom",
          strip.background = element_rect(fill = 'white', colour = 'white'))
}
ggplot() +
  geom_raster(data = train.images.ave, aes(x = x, y = y, fill = pixel)) +
  my_theme() +
  labs(x = NULL, y = NULL, fill = "Average scaled pixel value") +
  ggtitle('Average image in Fashion MNIST training data')

# Obtain covariance matrix and principal components for training data
library(stats)
cov.train = cov(train.images)                  
pca.train = prcomp(cov.train)                  

# Obtain cumulative proportion of variance against the number of principal components
plotdf = data.frame(index = 1:(pixs^2), 
                    cumvar = summary(pca.train)$importance["Cumulative Proportion", ])
t(head(plotdf, 50)) 

# Plot cumulative proportion of variance against the number of principal components
ggplot() + 
  geom_point(data = plotdf, aes(x = index, y = cumvar), color = "red") +
  labs(x = "Index of primary component", y = "Cumulative proportion of variance") +
  my_theme() +
  theme(strip.background = element_rect(fill = 'white', colour = 'black'))

# Subset the first pca.dims principal components that explain at least 99.5% of the variance
pca.dims = which(plotdf$cumvar >= .995)[1] 
pca.rot = pca.train$rotation[, 1:pca.dims]       

# Multiply the image data by the rotation matrix to obtain transformed image data
train.images.pca = data.frame(as.matrix(train.images) %*% pca.rot)
test.images.pca  = data.frame(as.matrix(test.images) %*% pca.rot)

# Combine image data with labels
train.data.pca = cbind(train.images.pca, label = factor(train.data$label))
test.data.pca = cbind(test.images.pca, label = factor(test.data$label))

################################################################################################
###################################### MODEL PERFORMANCE #######################################
################################################################################################

# Function to return model performance metrics
model_performance = function(fit, trainX, testX, trainY, testY, model_name){
  
  # Predictions on train and test data for different types of models
  if (any(class(fit) == "rpart")){
    
    library(rpart)
    pred_train = predict(fit, newdata = trainX, type = "class")
    pred_test = predict(fit, newdata = testX, type = "class")
    
  } else if (any(class(fit) == "train")){
    
    library(data.table)
    pred_dt = as.data.table(fit$pred[, names(fit$bestTune)]) 
    names(pred_dt) = names(fit$bestTune)
    index_list = lapply(1:ncol(fit$bestTune), function(x, DT, tune_opt){
      return(which(DT[, Reduce(`&`, lapply(.SD, `==`, tune_opt[, x])), .SDcols = names(tune_opt)[x]]))
    }, pred_dt, fit$bestTune)
    rows = Reduce(intersect, index_list)
    pred_train = fit$pred$pred[rows]
    pred_test = predict(fit, newdata = testX)
    trainY = fit$pred$obs[rows]
    
  } else {
    
    print(paste0("Error: Function evaluation unknown for object of type ", class(fit)))
    break
    
  }
  
  # Performance metrics on train and test data
  library(MLmetrics)
  df = data.frame(accuracy_train = Accuracy(trainY, pred_train),
                  precision_train = Precision(trainY, pred_train),
                  recall_train = Recall(trainY, pred_train),
                  F1_train = F1_Score(trainY, pred_train), 
                  accuracy_test = Accuracy(testY, pred_test),
                  precision_test = Precision(testY, pred_test),
                  recall_test = Recall(testY, pred_test),
                  F1_test = F1_Score(testY, pred_test),
                  model = model_name)
  
  print(df)
  
  return(df)
}

################################################################################################
######################################### SINGLE TREES #########################################
################################################################################################

# Single tree
library(rpart)
set.seed(1234)
tree = rpart(label ~., method = "class", data = train.data.pca)
plotcp(tree)
printcp(tree)
mp.single.tree = model_performance(tree, train.images.pca, test.images.pca, 
                                   train.data.pca$label, test.data.pca$label, "single_tree")
save(tree, file = "saved_objects/single_tree_pca.Rdata")

# Pruned tree
set.seed(1234)
prune.tree = prune(tree, cp = tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"])
mp.single.tree = model_performance(prune.tree, train.images.pca, test.images.pca, 
                                   train.data.pca$label, test.data.pca$label, "pruned_tree")
save(prune.tree, file = "saved_objects/prune_tree_pca.Rdata")

# Plot single and pruned trees
par(mfrow = c(1, 2))
plot(tree, uniform = TRUE, main = "Classification Tree")
text(tree, cex = 0.5)
plot(prune.tree, uniform = TRUE, main = "Pruned Classification Tree")
text(prune.tree, cex = 0.5) 
par(mfrow = c(1, 1))

################################################################################################
######################################## RANDOM FORESTS ########################################
################################################################################################

# Random forest - random search for mtry acoss pca.dims values with 5-fold cross-validation
library(caret)
rf_rand_control = trainControl(method = "repeatedcv", 
                               search = "random", 
                               number = 5, 
                               repeats = 5, 
                               allowParallel = TRUE, 
                               savePredictions = TRUE)
set.seed(1234)
rf_rand = train(x = train.images.pca, 
                y = train.data.pca$label,
                method = "rf", 
                ntree = 200,
                metric = "Accuracy", 
                trControl = rf_rand_control, 
                tuneLength = pca.dims) 
save(rf_rand, file = "saved_objects/rf_rand_pca.Rdata")
print(rf_rand)
plot(rf_rand)
mp.rf.rand = model_performance(rf_rand, train.images.pca, test.images.pca, 
                               train.data.pca$label, test.data.pca$label, "random_forest_random")

# Random forest - grid search for mtry acoss values 1:pca.dims with 5-fold cross-validation
rf_grid_control = trainControl(method = "repeatedcv", 
                               search = "grid", 
                               number = 5, 
                               repeats = 5, 
                               allowParallel = TRUE, 
                               savePredictions = TRUE)
set.seed(1234)
rf_grid = train(x = train.images.pca, 
                y = train.data.pca$label,
                method = "rf", 
                ntree = 200,
                metric = "Accuracy", 
                trControl = rf_grid_control,
                tuneGrid = expand.grid(.mtry = c(1:pca.dims)))
save(rf_grid, file = "saved_objects/rf_grid_pca.Rdata")
print(rf_grid)
plot(rf_grid)
mp.rf.grid = model_performance(rf_grid, train.images.pca, test.images.pca, 
                               train.data.pca$label, test.data.pca$label, "random_forest_grid")

# Obtain best model with highest accuracy
rf_models = list(rf_rand$finalModel, rf_grid$finalModel)
rf_accs = unlist(lapply(rf_models, function(x){ sum(diag(x$confusion)) / sum(x$confusion) }))
rf_best = rf_models[[which.max(rf_accs)]]

# Plot forest size versus error and variable importance for best model
library(randomForest)
plot(rf_best, main = "Relation between error and random forest size")
varImpPlot(rf_best)

# Obtain predicitons and confustion matrix from best model
library(reshape2)
pred = predict(rf_best, test.images.pca, type = "class")
conf = table(true = cloth_cats[as.numeric(test.data.pca$label)], 
             pred = cloth_cats[as.numeric(pred)])
conf = data.frame(conf / rowSums(conf))

# Plot the confusion matrix for a visual representation of model performance per class
ggplot() + 
  geom_tile(data = conf, aes(x = true, y = pred, fill = Freq)) + 
  labs(x = "Actual", y = "Predicted", fill = "Proportion") +
  my_theme() +
  scale_fill_continuous(breaks = seq(0, 1, 0.25)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  coord_fixed()

# Obtain required data to plot the ROC curves
library(ROCR)
library(plyr)
pred_roc = predict(rf_best, test.images.pca, type = "prob")
classes = unique(test.data.pca$label)
classes = classes[order(classes)]
plot_list = list()
for (i in 1:length(classes)) { 
  actual = ifelse(test.data.pca$label == classes[i], 1, 0)
  pred = prediction(pred_roc[, i], actual)
  perf = performance(pred, "tpr", "fpr")
  plot_list[[i]] = data.frame(matrix(NA, nrow = length(perf@x.values[[1]]), ncol = 2))
  plot_list[[i]]['x'] = perf@x.values[[1]]
  plot_list[[i]]['y'] = perf@y.values[[1]]
}
plotdf = rbind.fill(plot_list)
plotdf["Class"] = rep(cloth_cats, unlist(lapply(plot_list, nrow)))

# Plot the ROC curves
ggplot() +
  geom_line(data = plotdf, aes(x = x, y = y, color = Class)) + 
  labs(x = "False positive rate", y = "True negative rate", color = "Class") +
  ggtitle("ROC curve per class") + 
  theme(legend.position = c(0.85, 0.35)) +
  coord_fixed() + 
  my_theme()

################################################################################################
########################################### BOOSTING ###########################################
################################################################################################

# Boosting - grid search for parameters with 5-fold cross-validation
xgb_control = trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  allowParallel = TRUE,
  savePredictions = TRUE
)

xgb_grid = expand.grid(
  nrounds = c(50, 100), 
  max_depth = seq(5, 15, 5),
  eta = c(0.002, 0.02, 0.2),
  gamma = c(0.1, 0.5, 1.0), 
  colsample_bytree = 1, 
  min_child_weight = c(1, 2, 3),
  subsample = c(0.5, 0.75, 1)
)

set.seed(1234)
xgb_tune = train(x = train.images.pca, 
                 y = train.classes,
                 method = "xgbTree",
                 trControl = xgb_control,
                 tuneGrid = xgb_grid
)
xgb_tune
save(xgb_tune, file = "saved_objects/xgb_tune_pca.Rdata")

# Model performance metrics and results for boosting
xgb_tune$results[which.max(xgb_tune$results$Accuracy), ]
mp.xgb = model_performance(xgb_tune, train.images.pca, test.images.pca, 
                           train.classes, test.classes, "xgboost")

# Display the confusion matrix for the test data
table(pred = predict(xgb_tune, test.images.pca),
      true = test.classes)

################################################################################################
#################################### SUPPORT VECTOR MACHINE ####################################
################################################################################################

# Radial SVM - random search for C using 5-fold cross-validation and multi-class classification
library(MLmetrics)
svm_control = trainControl(method = "repeatedcv",   
                           number = 5,  
                           repeats = 5, 
                           classProbs = FALSE,
                           allowParallel = TRUE, 
                           summaryFunction = multiClassSummary,
                           savePredictions = TRUE)

set.seed(1234)
svm_rand_radial = train(label ~ ., 
                         data = cbind(train.images.pca, label = train.classes),
                         method = "svmRadial", 
                         trControl = svm_control, 
                         tuneLength = pca.dims,
                         metric = "Accuracy")
svm_rand_radial
save(svm_rand_radial, file = "saved_objects/svm_rand_radial_pca.Rdata")
mp.svm.rand.radial = model_performance(svm_rand_radial, train.images.pca, test.images.pca, 
                                       train.classes, test.classes, "svm_random_radial")

# Obtain predictions to plot confusion matrix for svm_rand_radial
library(data.table)
pred_dt = as.data.table(svm_rand_radial$pred[, names(svm_rand_radial$bestTune)]) 
names(pred_dt) = names(svm_rand_radial$bestTune)
index_list = lapply(1:ncol(svm_rand_radial$bestTune), function(x, DT, tune_opt){
  return(which(DT[, Reduce(`&`, lapply(.SD, `==`, tune_opt[, x])), .SDcols = names(tune_opt)[x]]))
}, pred_dt, svm_rand_radial$bestTune)
rows = Reduce(intersect, index_list)
pred_train = svm_rand_radial$pred$pred[rows]
trainY = svm_rand_radial$pred$obs[rows]
conf = table(pred_train, trainY)

# Plot the confusion matrix in a tile plot
conf = data.frame(conf / rowSums(conf))
ggplot() + 
  geom_tile(data = conf, aes(x = trainY, y = pred_train, fill = Freq)) + 
  labs(x = "Actual", y = "Predicted", fill = "Proportion") +
  my_theme() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_fill_continuous(breaks = seq(0, 1, 0.25)) +
  coord_fixed()

# Radial SVM - grid search for C and sigma using 5-fold cross-validation and multi-class classification
svm_grid_radial = expand.grid(sigma = c(.01, 0.04, 0.1), C = c(0.01, 10, 32, 70, 150))
set.seed(1234)
svm_grid_radial = train(label ~ ., 
                         data = cbind(train.images.pca, label = train.classes),
                         method = "svmRadial", 
                         trControl = svm_control, 
                         tuneGrid = svm_grid_radial,
                         metric = "Accuracy")
svm_grid_radial
save(svm_grid_radial, file = "saved_objects/svm_grid_radial_pca.Rdata")
mp.svm.grid.radial = model_performance(svm_grid_radial, train.images.pca, test.images.pca, 
                                       train.classes, test.classes, "svm_grid_radial")

# Plot accuracy as a function of cost, lines colored by sigma
ggplot() + 
  my_theme() +
  geom_line(data = svm_grid_radial$results, aes(x = C, y = Accuracy, color = factor(sigma))) +
  geom_point(data = svm_grid_radial$results, aes(x = C, y = Accuracy, color = factor(sigma))) +
  labs(x = "Cost", y = "Cross-Validation Accuracy", color = "Sigma") +
  ggtitle('Relationship between cross-validation accuracy and values of cost and sigma')

# Linear SVM - grid search for C using 5-fold cross-validation and multi-class classification
svm_grid_linear = expand.grid(C = c(1, 10, 32, 75, 150))
set.seed(1234)
svm_grid_linear = train(label ~ ., 
                         data = cbind(train.images.pca, label = train.classes),
                         method = "svmLinear", 
                         trControl = svm_control, 
                         tuneGrid = svm_grid_linear,
                         metric = "Accuracy")
svm_grid_linear
save(svm_grid_linear, file = "saved_objects/svm_tune_grid_linear_pca.Rdata")
mp.svm.grid.linear = model_performance(svm_grid_linear, train.images.pca, test.images.pca, 
                                       train.classes, test.classes, "svm_grid_linear")

# Boxplots of resampled accuracy for linear and radial Kernel SVMs
resamp_val = resamples(list(svm_radial = svm_grid_radial, svm_linear = svm_grid_linear))
plotdf = data.frame(Accuracy = c(resamp_val$values$`svm_radial~Accuracy`, resamp_val$values$`svm_linear~Accuracy`),
                    Model = rep(c("Radial Kernel", "Linear Kernel"), rep(nrow(resamp_val$values), 2)))
ggplot() +
  geom_boxplot(data = plotdf, aes(x = Model, y = Accuracy)) +
  ggtitle('Resample accuracy for SVM with linear and radial Kernel') + 
  my_theme()

################################################################################################
#################################### WRAPPING EVERYTHING UP ####################################
################################################################################################

# Combine all model performance metrics for all models to obtain overview
mp.df = rbind(mp.single.tree, mp.pruned.tree, mp.rf.rand, mp.rf.grid, mp.xgb, 
              mp.svm.rand.radial, mp.svm.grid.radial, mp.svm.grid.linear)
mp.df[order(mp.df$accuracy_test, decreasing = TRUE), ]
