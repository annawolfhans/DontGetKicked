library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(embed)
library(ranger)
library(discrim)
library(naivebayes)
library(kernlab)
library(themis)

kickTrain <- vroom("./training.csv", "NA", "NULL")
kickTest <- vroom("./test.csv",  "NA", "NULL")

# kickTrain <- kickTrain %>%
#   mutate(IsBadBuy = as.factor(IsBadBuy))

kickTrain <- kickTrain %>%
  mutate(IsBadBuy = as.factor(IsBadBuy)) %>%
  select(-VehYear, -BYRNO, -WheelTypeID, -VNST, -VNZIP1, 
         -PurchDate, -IsOnlineSale, -Trim, -Model, -SubModel, -AUCGUART, -PRIMEUNIT)

## Remove redundant variables 
# kickTrain <- kickTrain %>%
#   filter(-)




my_recipe <- recipe(IsBadBuy~., data=kickTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_select(-VehYear, -BYRNO, -WheelTypeID, -VNST, -VNZIP1, 
              -PurchDate, -IsOnlineSale, -Trim, -AUCGUART, -PRIMEUNIT) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.8) %>%
  step_smote(all_outcomes(), neighbors=5)

  
prep <- prep(my_recipe)
bakedTest <- bake(prep, new_data=kickTest)
bakedTrain <- bake(prep, new_data=kickTrain)

#### RANDOM FORESTS ####
my_mod <- rand_forest(mtry=tune(),
                      min_n=tune(),
                      trees=100) %>%
  set_engine("ranger") %>%
  set_mode("classification")


kick_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=kickTrain)  


tuning_grid <- grid_regular(mtry(range=c(1,1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(kickTrain, v=5, repeats=1)

CV_results <- kick_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))



# metric_set(roc_auc, f_meas, sens, recall, spec, 
# precision, accuracy)

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- kick_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=kickTrain)

# final_wf %>%  
#   predict(new_data=amazonTest, type="prob")

kick_predictions <- predict(final_wf,
                              new_data=kickTest,
                              type="prob") %>%
  bind_cols(., kickTest) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy=.pred_1)

vroom_write(x=kick_predictions, file="./KickClassificationRF.csv", delim=",")


#### PENALIZED LOGISTIC REGRESSION ####
library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(embed)

kickTrain <- vroom("./train.csv")
kickTest <- vroom("./test.csv")

kickTrain <- kickTrain %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
  set_engine("glmnet")

# my_recipe <- recipe(ACTION~., data=amazonTrain) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_other(all_nominal_predictors(), threshold=0.001) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

my_recipe <- recipe(IsBadBuy~., data=kickTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  #step_dummy(all_nominal_predictors()) %>%
  #step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_normalize(all_predictors()) %>%
  # step_pca(all_predictors(), threshold = 0.8) %>%
  step_smote(all_outcomes(), neighbors=5)

# prep <- prep(my_recipe)
# baked <- bake(prep, new_data=amazonTest)

final_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=kickTrain)
#
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels=5)

folds <- vfold_cv(amazonTrain, v=5, repeats=1)

CV_results <- final_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))


# metric_set(roc_auc, f_meas, sens, recall, spec,
#   precision, accuracy)

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- final_workflow %>%
  finalize_workflow(bestTune)

# final_wf %>%
#   predict(new_data=amazonTest, type="prob")

kick_predictions <- predict(final_wf,
                              new_data=kickTest,
                              type="prob") %>%
  bind_cols(., kickTest) %>%
  select(id, .pred_1) %>%
  rename(Action=.pred_1)

vroom_write(x=kick_predictions, file="./KickSMOTEPenalizedPreds.csv", delim=",")
