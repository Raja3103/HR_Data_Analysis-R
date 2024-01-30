library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)

hr_test= read.csv(r'(D:\Raja\R\Project 4 - Human Resources\hr_test.csv)', stringsAsFactors = F)
hr_train= read.csv(r'(D:\Raja\R\Project 4 - Human Resources\hr_train.csv)',  stringsAsFactors = F)


HR = hr_train
HR_t = hr_test

head(HR)
dim(HR)
view(HR)


## as.numeric function
numeric_f <- function(x){
  return(as.numeric(as.character(x)))
}

## changing numeric to character type

HR$Work_accident = as.character(HR$Work_accident)
HR$promotion_last_5years = as.character(HR$promotion_last_5years)
HR$left = as.factor(HR$left)

## creating pipe and preparing data 

bx_pipe = recipe(left ~ . , data = HR) %>%
  update_role(last_evaluation,satisfaction_level,number_project,average_montly_hours,time_spend_company,new_role = 'to_numeric') %>%
  update_role(Work_accident,promotion_last_5years,sales,salary,new_role = 'to_dummy') %>%
  step_unknown(has_role('to_dummy'), new_level = '__missing__') %>%
  step_other(has_role('to_dummy'), threshold = 0.05,other = '__other__') %>% 
  step_dummy(has_role('to_dummy')) %>% 
  step_mutate_at(has_role('to_numeric'),fn= numeric_f) %>%
  step_impute_median(all_numeric(),-all_outcomes())




bx_pipe = prep(bx_pipe)


H_train = bake(bx_pipe, new_data = NULL)
H_test = bake(bx_pipe, new_data = HR_t)


set.seed(2)
s= sample(1:nrow(H_train),0.8*nrow(H_train))
t1 = H_train[s,] ##80% of train data 
t2 = H_train[-s,] ## 20% of train data for validation


## creating RF model 
rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

## creating 10 folds

folds = vfold_cv(H_train, v=7)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)), 
                       min_n(c(2,10)), levels = 3)


my_res=tune_grid(
  rf_model,
  left~.,
  resamples = folds,
  grid = rf_grid,
  metrics = yardstick::metric_set(yardstick::roc_auc),
  control = control_grid(verbose = TRUE)
)


## looking for plots against hyperparameters
autoplot(my_res)+theme_light()

## collecting all the average value of roc_auc against each of the folds per hyperparameter 
fold_metrics=collect_metrics(my_res)

my_res %>% show_best()


final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(left~.,data=H_train)


# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = t1,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = t2,type="prob") %>% select(.pred_1)


train.score= train_pred$.pred_1
test.score = test_pred$.pred_1

real_train = t1$left
real_test = t2$left
## checking for auc/roc

caTools::colAUC(train.score,real_train, plotROC = T)
caTools::colAUC(test.score,real_test, plotROC = T)




test_pred_final=predict(final_rf_fit,new_data = H_test,type="prob") %>% select(.pred_1)

write.csv(test_pred_final, "Raja_Barman_P4_part2.csv", row.names = F)





