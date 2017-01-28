# Weight Lifting Exercise Data Analysis
Svetlana Aksyuk (s.a.aksuk@gmail.com)  
27 Jan 2017  
  


  
## Task
This report is a course project from the Practical Machine Learning by Jeff Leek and the Data Science Track Team. The data for this project was collected and first investigated by: Velloso, Bulling, Gellersen, Ugulino & Fuks (2013). In order to apply knowledge about basic machine learning techniques, we need to train a model to classify observations of physical exercises (class A corresponds to the correct performance, while common mistakes are coded as classes B-E).    

## Data Exploration  
  

  
Dataset contains 160 variables. First, raw training dataset was splitted in two parts: one for model training (75%, or 14718 observations), another for validation of results (the remaining 25%, or 4904). Testing set contains 20 observations.   
  

  
### Non-numeric variables
  
Dependent variable is named ```classe```. Proportion of class A is about 0.28 of the training set. Classes B, C, D, E are roughly around 0.18 (**Figure** <a href="#Plot-01">1</a>-A). **Table** <a href="#Table-01">table 1</a> shows  variables, which values are neither numeric nor integer. As we can see, all logical variables are empty, so can be removed. We need to run some tests to determine whether ```classe``` depends on character variables ```user_name``` and ```new_window```. Variable ```cvtd_timestamp``` has to be recoded into data format and to be treated as numeric.  
  
  Theoretically, there should be no connection between individual who performed the task and dependent variable. But P-value of chi-square test for independence is 0.0000, which is less than 0.05, so this variables are somehow connected. We can see from **Figure **<a href="#Plot-01">1</a>-B, that histograms of ```classe``` for different users are not the same. Nevertheless, the question is: can we predict whether exercise is performed correctly, using data from sensors for any individual? To answer this, we have to eliminate information about user from the training set. For the same reasons we exclude ```cvtd_timestamp```, though chi-square test rejects hypothesis about independence (P-value = 0.0000). We drop ```raw_timestamp_part_1``` and ```raw_timestamp_part_2``` too.    
  
  Variable ```new_window``` is, in fact, logical (**Figure **<a href="#Plot-01">1</a>-C) and represents some characteristic of an experiment. Since P-value of chi-square test for independence: 0.7985 is greater than 0.05, we can exlude this factor from the dataset.  
  
<a name="Plot-01"></a><div class="figure" style="text-align: center">
<img src="WLE_analysis_files/figure-html/Plot-01-1.png" alt="**Figure ** 1: Proportions of classes in the training set"  />
<p class="caption">**Figure ** 1: Proportions of classes in the training set</p>
</div>

<a name="Table-01"></a>

Table: **Table ** 1: Non numeric (integer) variables

Class       Variables                                                                                                                         NA.percent
----------  -------------------------------------------------------------------------------------------------------------------------------  -----------
character   user_name, cvtd_timestamp, new_window                                                                                                      0
factor      classe                                                                                                                                     0
logical     kurtosis_yaw_belt, skewness_yaw_belt, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm           100




### Numeric variables



<a name="Table-02"></a>

Table: **Table ** 2: Percent of NA in numeric (integer) variables

variables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          na.percent
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  -----------
kurtosis_roll_belt, skewness_roll_belt, max_roll_belt, max_picth_belt, max_yaw_belt, min_roll_belt, min_pitch_belt, min_yaw_belt, amplitude_roll_belt, amplitude_pitch_belt, amplitude_yaw_belt, var_total_accel_belt, avg_roll_belt, stddev_roll_belt, var_roll_belt, avg_pitch_belt, stddev_pitch_belt, var_pitch_belt, avg_yaw_belt, stddev_yaw_belt, var_yaw_belt, var_accel_arm, avg_roll_arm, stddev_roll_arm, var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm, kurtosis_yaw_arm, skewness_yaw_arm, max_roll_arm, max_picth_arm, max_yaw_arm, min_roll_arm, min_pitch_arm, min_yaw_arm, amplitude_roll_arm, amplitude_pitch_arm, amplitude_yaw_arm, kurtosis_roll_dumbbell, kurtosis_picth_dumbbell, skewness_roll_dumbbell, skewness_pitch_dumbbell, max_roll_dumbbell, max_picth_dumbbell, max_yaw_dumbbell, min_roll_dumbbell, min_pitch_dumbbell, min_yaw_dumbbell, amplitude_roll_dumbbell, amplitude_pitch_dumbbell, amplitude_yaw_dumbbell, var_accel_dumbbell, avg_roll_dumbbell, stddev_roll_dumbbell, var_roll_dumbbell, avg_pitch_dumbbell, stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell, stddev_yaw_dumbbell, var_yaw_dumbbell, max_roll_forearm, max_picth_forearm, min_roll_forearm, min_pitch_forearm, amplitude_roll_forearm, amplitude_pitch_forearm, var_accel_forearm, avg_roll_forearm, stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm          98.0
kurtosis_picth_belt, skewness_roll_belt.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                98.1
kurtosis_roll_arm, kurtosis_picth_arm, skewness_roll_arm, skewness_pitch_arm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             98.3
kurtosis_roll_forearm, kurtosis_picth_forearm, skewness_roll_forearm, skewness_pitch_forearm, max_yaw_forearm, min_yaw_forearm, amplitude_yaw_forearm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    98.4


  
**Figure ** <a href="#Plot-02">2</a> shows plots of first numeric variables with maximum standard deviation. There are variable ```X```, which separates classes perfectly. Since we could not find out the meaning of this variable, and the goal is to train a model on multivariate data, we had excluded ```X``` from dataset too. Next step is to train and compare models using 53 independent variables.  
  
<!-- This chunk is cached -->
<a name="Plot-02"></a><div class="figure" style="text-align: center">
<img src="WLE_analysis_files/figure-html/Plot-02-1.png" alt="**Figure ** 2: First four numeric variables by class"  />
<p class="caption">**Figure ** 2: First four numeric variables by class</p>
</div>
  

  
<!-- This chunk is cached -->

  


### Data transformations and choice of classification method
  
All remaining independent variables are numeric. Correlations between them were estimated with Spearman coefficient, and 89.3% of coefficients are highly significant (P-values are less than 0.01); about 1.7% of them show high correlations (absolute values of coefficient are greater than 0.7). Spearman coefficient also evaluates nonlinear relationships, which makes it more universal estimator, than Pearson correlations.  
  
Some models are sensitive to the correlation of factors, so we will try two approaches of data transformation:  
  
* 1: standardised data (mean = 0, sd = 1);  
* 2: principal components, standardised and uncorrelated, which describe at least 90% of variance (19 PC at training dataset);  
* 3: first twelve principal components (the number was determined empirically as ```min(12, 15)```, see **Figure **<a href="#Plot-03">3</a>).  
  
Since this is a case of supervised learning, we compared five types of models at principal components without cross validation:  

1. Classification tree -- CART (```rpart()``` from package ```rpart```).  
2. Random forest (```randomForest()```, package ```randomForest```).  
3. Linear discriminant analysis -- LDA (```lda()```, package ```MASS```).  
4. Quadratic discriminant analysis -- QDA (```qda()```, package ```MASS```).  
5. K-nearest neighbour classification with k = 3 and k = 6 (```knn()```, package ```class```).
  
After examination of model errors based on overral accuracy (**Figure **<a href="#Plot-03">3</a>), two models with errors less than 3% were chosen for training:   

* A: k-nearest neighbour classification;    
* B: random forest.  
  
<!-- This chunk is cached -->



  
![](p1.png) ![](p2.png) ![](p3.png)
  
<a name="Plot-03, "></a><div class="figure" style="text-align: center">
<img src="x.png" alt="**Figure ** 3: Overral validation set errors for different numbers of PC, without cross validation" width="480" />
<p class="caption">**Figure ** 3: Overral validation set errors for different numbers of PC, without cross validation</p>
</div>

## Models comparison
  
<!-- This chunk is cached -->


<!-- This chunk is cached -->


<!-- This chunk is cached -->


<!-- This chunk is cached -->


<!-- This chunk is cached -->


<!-- This chunk is cached -->

  
For more precise calculation of prediction errors we use k-fold validation with 3 folds in each model at this step via ```train()``` function from package ```caret```. **Table **<a href="#Table-03">3</a> shows accuracy of six models in comparison (3 data transformation approaches and 2 model types), estimated for validation sample. Random forest performs better than k-nearest neighbour classification. Random forest on standardized data shows perfect accuracy, which definitely looks like overfitting. First 19 principal components explain more than 90% of variation of 53 independent variables. Further reduction of the number of components to 12 results in a slight decrease in the accuracy of models.   
  
**Considering accuracy and minimum number of predictors, the best model is the latest: random forest on 12 principal components.**  
  
<a name="Table-03"></a>

Table: **Table ** 3: Accuracy of models calculated on validation set

data             Method          Overral.Accuracy   Min.Balanced.Accuracy   Max.Balanced.Accuracy 
---------------  --------------  -----------------  ----------------------  ----------------------
Standardised     KNN, k = 5      0.97               Class: C; 0.97          Class: E; 0.99        
PC explain 90%   KNN, k = 5      0.96               Class: C; 0.96          Class: E; 0.99        
12 PC            KNN, k = 5      0.95               Class: C; 0.94          Class: E; 0.99        
Standardised     Random forest   1                  Class: C; 1             Class: E; 1           
PC explain 90%   Random forest   0.98               Class: C; 0.97          Class: E; 1           
12 PC            Random forest   0.97               Class: C; 0.97          Class: E; 0.99        

## Prediction and out of sample error estimation
  
Out of sample error, estimated as (1 - Overral Accuracy)\*100%, equals to **3%** for the best model.  
  
Predictions for the testing set are listed below. Predictions made using the best k-nearest neighbour model (model #1) are the same except for one observation.  
  

```r
# transform testing sample
preObj <- preProcess(training.set[, !(names(training.set) %in% 'classe')],
                     method = 'pca', pcaComp = num.pca)
test.data <- predict(preObj, testing)
# use the best model to make prediction 
best.prediction <- predict(model.rf.3, newdata = test.data)
names(best.prediction) <- 1:20
# show results
print(best.prediction)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  A  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
  
Results for the best model were used as answers to the test, and it appears, that observation 3 has not been classified correctly by the model 6. Prediction for the third observation with model 1 (best KNN) is also incorrect. Model number four, random forest on standardized data, gives correct answer (it seems that accuracy = 1 does not necessarily mean overfitting):   


```r
# correct predictions
preObj <- preProcess(training.set[, !(names(training.set) %in% 'classe')],
                     method = c('center', 'scale'))
test.data <- predict(preObj, testing)
correct.prediction <- predict(model.rf.1, newdata = test.data)
names(correct.prediction) <- 1:20
print(correct.prediction)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
  


## References

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
