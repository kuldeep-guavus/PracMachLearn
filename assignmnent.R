
pml_training <- read.csv("./pml-training.csv")
pml_testing <- read.csv("./pml-testing.csv")

pml_split <- createDataPartition(pml_training$classe, p=0.7, list=FALSE)
validation <- pml_training[-pml_split,]
work_data <- pml_training[pml_split,]
pml_train_split <- createDataPartition(work_data$classe, p=0.7, list=FALSE)
pml_train <- work_data[pml_train_split,]
pml_test <- work_data[-pml_train_split,]

pml_nzv <- nearZeroVar(pml_train, saveMetrics=TRUE)
pml_no_nzv <- subset(pml_nzv, nzv %in% c(FALSE))
pml_no_nzv_names <- rownames(pml_no_nzv)
pml_train_nz <- pml_train[,!(colnames(pml_train) %in% pml_no_nzv_names)]

pml_vars <- data.frame(lapply(pml_train_nz, var)) 
pml_vars_nz <- pml_vars[ , unlist(lapply(pml_vars, function(x) !all(is.na(x))))]
pml_final_names <- colnames(pml_vars_nz)

pml_train_final <- pml_train_nz[,(colnames(pml_train_nz) %in% pml_final_names)]
pml_train_final <- subset(pml_train_final, select=-c(user_name, cvtd_timestamp))

pml_pca_model <- preProcess(pml_train_final[,-57], method="pca", thresh=0.8)
pca_comps <- predict(pml_pca_model, pml_train_final[,-57])
pca_comps$classe <- pml_train_final$classe

pml_test_nz <- pml_test[,(colnames(pml_test) %in% pml_no_nzv_names)]
pml_test_final <- pml_test_nz[,(colnames(pml_test_nz) %in% pml_final_names)]
pml_test_final <- pml_test_final[,-c(user_name,cvtd_timestamp)]

pml_pca_rf_model <- train(classe ~ .,method="rf",data=pca_comps)
pml_pca_gbm_model <- train(classe ~ .,method="gbm",data=pca_comps,verbose=FALSE)
pml_pca_tree_bag_model <- train(classe ~ .,method="treebag",data=pca_comps)

pca_comps_test <- predict(pml_pca_model, pml_test_final[,-57])
pca_test_pred <- predict(pml_pca_gbm_model, pca_comps_test)
confusionMatrix(pml_test_final$classe, predict(pml_pca_rf_model, pca_comps_test))
confusionMatrix(pml_test_final$classe, predict(pml_pca_gbm_model, pca_comps_test))
confusionMatrix(pml_test_final$classe, predict(pml_pca_tree_bag_model, pca_comps_test))

pml_testing_nz <- pml_testing[,(colnames(pml_testing) %in% pml_no_nzv_names)]
pml_testing_final <- pml_testing_nz[,(colnames(pml_testing_nz) %in% pml_final_names)]
pml_testing_final <- subset(pml_testing_final, select=-c(user_name, cvtd_timestamp))
pca_comps_testings <- predict(pml_pca_model, pml_testing_final)

predict(pml_pca_tree_bag_model, pca_comps_testings)
