# Diagnostic Prediction - Breast Cancer 
**Authors: Jennifer Jiang, Vivienne Yu**

## 1. Abstract
Breast cancer ranks as the second most prevalent cancer among women in the United States. The current diagnostic process is often protracted, spanning several weeks. In an effort to expedite this critical diagnostic phase, the study have explored the application of machine learning models. It utilizes a comprehensive dataset comprising attributes of breast masses, derived from breast tumors, which may or may not be cancerous. Through rigorous examination of various subset selection models, the most impactful predictors are identified for the 'diagnosis' outcome. These key variables are associated with concavity, compactness, radius, texture, area, and smoothness. Subsequently, the study assessed a range of machine learning models to determine the one yielding the highest accuracy. The findings reveal that the random forest model excels in predicting diagnoses, thereby offering a promising approach for enhancing the efficiency and accuracy of breast cancer diagnosis.

## 2. Introduction
### 2.1 Problems and Motivations
This study aims to diagnose breast cancer based on specific features of a breast mass extracted from tumors, addressing two primary questions. The first concerns identifying the most determinative features of a breast mass from a tumor in breast cancer diagnosis. The second question investigates which machine learning model yields the highest accuracy for such diagnostic prediction. This inquiry is pertinent given that breast cancer is the second most common cancer among women in the United States, with approximately 13% (about 1 in 8) of U.S. women expected to develop invasive breast cancer during their lifetime. Notably, for a prevalent cancer like this, the diagnostic process for different types (malignant & benign) of breast tumors typically involves a series of tests and evaluations, potentially extending up to four weeks. Consequently, the study posits that employing machine learning techniques could significantly reduce the diagnosis timeframe, thereby facilitating earlier treatment for cancer patients.

### 2.2 Approaches
Since the study investigates two distinct problems, it will bifurcate and approach them uniquely. To determine the most critical features in the breast cancer diagnosis process, the study will employ forward, backward selection, and lasso techniques. The aim is to compare whether the results derived from each method align or differ. Additionally, the study will juxtapose these findings with the ranked important features as identified by the Random Forest model. The selection of these methods is informed by their academic exploration in class, coupled with the belief that they will likely yield similar features of importance.

To identify the most effective machine learning models for diagnosing breast cancer, this study will evaluate a diverse array of models. These include KNN (K-Nearest Neighbors) classification, logistic regression (logit model), decision tree classification, random forest, and KMeans clustering, which will be included in the supplementary approach section. The logistic regression model is a parametric model, while KNN classification, decision tree classification, and random forest are non-parametric models. Parametric models are typically more suited to smaller datasets due to their reliance on predefined forms, whereas non-parametric models excel in handling high-dimensional data due to their flexibility in model structure. This contrast makes it intriguing to compare the outcomes from both model types.

According to the research "Prediction of Breast Cancer using Machine Learning Approaches" conducted by the National Center for Biotechnology Information, the random forest model was identified as the most accurate among all tested models. Consequently, it is anticipated that the random forest model will also demonstrate superior performance in this study.

## 3. Setup
### 3.1 Dataset
The dataset used for this research is called the Breast Cancer Wisconsin (Diagnostic) Dataset collected and published by UCI Machine Learning Repository. The dataset contains a total of 30 different features that are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass extracted from the tumor, and they describe characteristics of the cell nuclei present in the image below. 
![Fine needle aspirate of a breast mass](https://github.com/Vivsquared/QTM-347-Final-Project/blob/main/FNA%20image.png)

The dataset features a categorical column labeled 'diagnosis', where 'M' denotes malignant (cancerous tumors) and 'B' indicates benign (non-cancerous tumors). For analytical purposes, this variable will be transformed into a binary format, with 1 representing malignant and 0 representing benign. Additionally, the dataset includes a column that is empty and another containing patient IDs; both of these will be omitted from the analysis to ensure data integrity and privacy. The table below enumerates all 30 features along with the response variable, diagnosis.

| Variable Name          | Description                                                     | Data Type    |
|------------------------|-----------------------------------------------------------------|--------------|
| diagnosis              | The diagnosis of breast tissues (M = malignant, B = benign)     | Categorical  |
| radius_mean            | mean of distances from center to points on the perimeter        | Numeric      |
| texture_mean           | standard deviation of gray-scale values                         | Numeric      |
| perimeter_mean         | mean size of the core tumor                                     | Numeric      |
| area_mean              | mean area of the core tumor                                     | Numeric      |
| smoothness_mean        | mean of local variation in radius lengths                       | Numeric      |
| compactness_mean       | mean of perimeter^2 / area - 1.0                                | Numeric      |
| concavity_mean         | mean of severity of concave portions of the contour             | Numeric      |
| concave points_mean    | mean for number of concave portions of the contour              | Numeric      |
| symmetry_mean          | mean of the symmetryness                                        | Numeric      |
| fractal_dimension_mean | mean for "coastline approximation" - 1                          | Numeric      |
| radius_se            | standard error for the mean of distances from center to points on the perimeter        | Numeric      |
| texture_se           | standard error for standard deviation of gray-scale values                         | Numeric      |
| perimeter_se         | standard error for mean size of the core tumor                                     | Numeric      |
| area_se              | standard error for mean area of the core tumor                                     | Numeric      |
| smoothness_se        | standard error for local variation in radius lengths                       | Numeric      |
| compactness_se       | standard error for perimeter^2 / area - 1.0                                | Numeric      |
| concavity_se         | standard error for severity of concave portions of the contour             | Numeric      |
| concave points_se    | standard error for number of concave portions of the contour              | Numeric      |
| symmetry_se          | standard error the symmetryness                                        | Numeric      |
| fractal_dimension_se | standard error for "coastline approximation" - 1                          | Numeric      |
| radius_worst            | "worst" or largest mean value for mean of distances from center to points on the perimeter        | Numeric      |
| texture_worst           | "worst" or largest mean value for standard deviation of gray-scale values                         | Numeric      |
| perimeter_worst         | "worst" or largest mean value for size of the core tumor                                     | Numeric      |
| area_worst              | "worst" or largest mean value for area of the core tumor                                     | Numeric      |
| smoothness_worst        | "worst" or largest mean value for local variation in radius lengths                       | Numeric      |
| compactness_worst       | "worst" or largest mean value for perimeter^2 / area - 1.0                                | Numeric      |
| concavity_worst         | "worst" or largest mean value for severity of concave portions of the contour             | Numeric      |
| concave points_worst    | "worst" or largest mean value for number of concave portions of the contour              | Numeric      |
| symmetry_worst          | "worst" or largest mean value for the symmetryness                                        | Numeric      |
| fractal_dimension_worst | "worst" or largest mean value for "coastline approximation" - 1                          | Numeric      |

There are a total of 569 observations. With 357 malignant and 212 benign.
![Distribution plot for diagnosis](https://github.com/Vivsquared/QTM-347-Final-Project/blob/main/Explorative%20Data%20Analysis/Distribution%20of%20Diagnosis.png)

This is a heatmap for all the mean variables and diagnosis. 
- 'radius_worst', 'perimeter_worst', 'area_worst', 'concave points_worst' have high correlations with 'diagnosis', suggesting that they are strong indicators for predicting the malignancy of a tumor.
- The features themselves are also correlated with each other. For example, 'radius_worst' has a very high correlation with 'perimeter_worst' and 'area_worst', which makes sense as larger radii typically lead to larger perimeters and areas.
![heatmap for all the mean variables](https://github.com/Vivsquared/QTM-347-Final-Project/blob/main/Explorative%20Data%20Analysis/Mean%20Variables%20Heatmap.png)

This is a heatmap for all the standard error variables and diagnosis. The correlations between standard error variables and diagnosis are weak. The standard error of area-related variables are highly correlated with each other. The standard error of compactness is also highly correlated with the standard error of concavity.
- 'radius_se' has a moderate positive correlation with diagnosis (0.57), which might suggest that larger standard error of the radius measurement is associated with malignancy. Conversely, 'texture_se' shows virtually no correlation with diagnosis (-0.01), indicating that the standard error of the texture measurement might not be a good predictor of malignancy.
- Values closer to zero or negative (blues) indicate a weaker relationship. For example, 'smoothness_se' has a slight negative correlation with diagnosis (-0.07), which suggests that as the standard error of the smoothness measurement increases, the likelihood of a malignant diagnosis decreases slightly.
- The heatmap also shows how each of the SE variables correlates with one another. For instance, 'perimeter_se' and 'area_se' have a very high positive correlation (0.94), which is expected because the perimeter and area measurements are related geometrically. 
![heatmap for all the standard error variables](https://github.com/Vivsquared/QTM-347-Final-Project/blob/main/Explorative%20Data%20Analysis/SE%20Variables%20Heatmap.png)

This is a heatmap for all the worst value variables and diagnosis. There are some strong correlations between worst value variables and diagnosis and with each others. radius_worst, perimeter_worst, area_worst, concave points_worst are strongly correlated with diagnosis.
- 'radius_mean' (0.73), 'perimeter_mean' (0.74), 'area_mean' (0.71), 'concavity_mean' (0.70), and 'concave points_mean' (0.78) have strong positive correlations with the diagnosis, suggesting that larger mean values of these variables are associated with a cancerous diagnosis ('M' for malignant).
- The heatmap shows high correlations between several pairs of features, such as 'radius_mean' with 'perimeter_mean' and 'area_mean', which is expected because these geometric measurements are related to each other.
- 'fractal_dimension_mean' has very low correlations with most other variables, suggesting it provides unique information not linearly related to the other measurements.
- 'fractal_dimension_mean' has a slight negative correlation with 'radius_mean' (-0.31), 'perimeter_mean' (-0.26), and 'area_mean' (-0.28), indicating that as the mean fractal dimension increases, the mean values of radius, perimeter, and area tend to decrease slightly.
![heatmap for all the worst value variables](https://github.com/Vivsquared/QTM-347-Final-Project/blob/main/Explorative%20Data%20Analysis/Worst%20Variables%20Heatmap.png)

### 3.2 Model & Parameter setup 

**All models are ran using Python 3 on the Jupyter Notebook environment.**

**For this study, the dataset will be splited into 70% training set and 30% testing set.**

<details>

<summary>Forward Selection</summary>
<br>Forward selection is a stepwise subset selection. The forward selection start with no predictors and add predictors to the model one at once. The stepwise selecion is devided into 4 steps: the first step is when there is no prdictors; the second step is by fitting in p models with one predictor and select the best model; the third and fourth step is by fitting in more predictors based on the predictor fitted in previously. In each step, best is defined as having smallest RSS/MSE or highest R squared. In general, forward stepwise selection is more applicable to high-dimensional settings.

<br>For this study, the ultimate predictors for forward selection is set to 13 in the end. 

</details>

<details>

<summary>Backward Selection</summary>
<br>Backward selection is a stepwise subset selection. The backward selection start with all predictors and take away predictors from the model one at once. The stepwise selecion is devided into 4 steps: the first step is when there is p prdictors; the second step is by fitting in p models with p-1 predictors and select the best model; the third and fourth step is by fitting in p-n predictors based on the predictors fitted in previously. In each step, best is defined as having smallest RSS/MSE or highest R squared. In general, backward stepwise selection is applicable to low-dimensional settings, and backward selection is not returning same result as forward selection.

<br>For this study, the ultimate predictors for backward selection is set to 13 in the end. 

</details>

<details>

<summary>Lasso</summary>
<br> Lasso stands for the least absolute shrinkage and selection operator. It has the capability to perform variable selection. When lambda is larger than or equal to 0, lasso tunes the hyper parameter. Lasso minimizes: 
$\sum_{i} (y_i - \beta_0 - \sum_{j} \beta_j \cdot x_{ij})^2 + \lambda \cdot \sum_{j} |\beta_j|$ . 
Lasso has a high shrinkage penalty, while for multiple parameters shrinkage penalty $\lambda$ does not apply to \beta_0. As $\lambda$ increases, lasso select less variables. Lasso path is considered as different coefficient values by varying $\lambda$.

<br> Based on the $\lambda$ selected by the lasso, the ultimate predictor used for lasso is 26 predictors based on the coefficience (considering coefficience shrink to 0). 
</details>

<details>

<summary>KNN Model</summary>
<br> K-Nearest Neighbors (KNN) classification is a non-parametric method. The output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k=1, then the object is simply assigned to the class of that single nearest neighbor. The KNN classifier assumes that similar things exist in close proximity. In other words, similar data points are near to each other in the feature space.

<br> Before performing the KNN model, all the data for the dependent variables will be scale using StandardScaler in preventing features with larger magnitudes from dominating the distance calculations. In terms of optimization, 10-fold cross validation is used to find the best optimal k. 

</details>

<details>

<summary>Logit Model</summary>
<br> Logit Model is a parametric statistical method that models the probability of a binary outcome based on one or more predictor variables. It is a type of regression analysis that is appropriate for predicting the outcome of a categorical dependent variable, particularly when the dependent variable is binary—meaning it has only two possible outcomes (in this case, "Malignant" vs "Benign"). The probability of the outcome is modeled using the logistic function, which is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1. 

<br> The threshold for the logit model used in this study is set to 0.5, meaning any value greater than 0.5 will set to 1, malignant, and any value smaller than 0.5 will set to 0, benign. 

</details>

<details>

<summary>Decision Tree Model</summary>
<br> Decision tree classification is a supervised machine learning algorithm used to categorize data into classes based on the values of input features. It takes the form of a tree structure. It consists of nodes, branches, and leaves, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node holds a class label. The topmost node in a decision tree is known as the root node. It is from this node that the dataset is divided into subsets, which then form the basis of the branches connected to the node. This process of dividing is based on a set of decision rules derived from the data attributes.

<br> For this study, the criterion of the decision tree model is set to entropy. It will be fitted without a max_depth parameter, then it will be pruned to be optimized.
</details>

<details>

<summary>Random Forest</summary>
<br> Random Forest classification is a supervised machine learning algorithm that builds upon the concept of decision tree classification by creating an ensemble of trees using bagging method to improve prediction accuracy and control over-fitting. When fitting the tree, a random subset of *m&ltp* predictors are use and it will lead to very different trees from each sample. Finally, the prediction for each tree is averaged to get the result.

<br> For this study, a random forest model is built and tested for different combination of parameters, 'n_estimators': [50, 100, 150, 200, 250],'max_features': range(1, 31), 'min_samples_split': [2, 5, 10, 15], to get the most optimized model and accurate predication.

</details>
  
## 4. Results
### 4.1 Main Indications of the Result
This study comprises two distinct sections to determine breast tumor types and the presence of breast cancer: subset selection, machine learning methodology with cross validation.

<br> The study employed two subset selection methods: forward selection and backward selection. 

Forward selection is typically used when the number of predictors exceeds the number of samples, whereas backward selection is preferred when the sample size surpasses the number of predictors. Given our dataset of approximately 500 samples and 30 predictors, backward selection was initially applied. However, due to the relatively large number of predictors and limited sample size, and considering our extensive set of covariates, forward selection was deemed more beneficial (Bursac, 2008). The effectiveness of predictors in subset selection was evaluated based on R-squared values and P-values, with a higher R-squared and a P-value under 0.05 as the criteria for effective predictor selection. Consequently, the predictors chosen for forward selection included: concave_points_worst, radius_worst, texture_worst, area_worst, smoothness_se, symmetry_worst, compactness_se, radius_se, fractal_dimension_worst, compactness_mean, concave_points_mean, concavity_worst, and concavity_se.
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              diagnosis   R-squared:                       0.769
Model:                            OLS   Adj. R-squared:                  0.763
Method:                 Least Squares   F-statistic:                     141.8
Date:                Wed, 13 Dec 2023   Prob (F-statistic):          9.50e-167
Time:                        04:08:49   Log-Likelihood:                 22.556
No. Observations:                 569   AIC:                            -17.11
Df Residuals:                     555   BIC:                             43.70
Df Model:                          13                                         
Covariance Type:            nonrobust                                         
===========================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------
concave points_worst       -1.1717      0.552     -2.123      0.034      -2.255      -0.088
const                       3.1568      0.180     17.494      0.000       2.802       3.511
radius_worst               -0.1334      0.015     -9.042      0.000      -0.162      -0.104
texture_worst              -0.0107      0.002     -5.857      0.000      -0.014      -0.007
area_worst                  0.0009      0.000      7.731      0.000       0.001       0.001
s0oothness_se             -21.8644      4.381     -4.991      0.000     -30.469     -13.260
sy00etry_worst             -0.7695      0.210     -3.668      0.000      -1.181      -0.357
co0pactness_se              0.6599      1.293      0.510      0.610      -1.880       3.200
radius_se                  -0.2917      0.069     -4.229      0.000      -0.427      -0.156
fractal_di0ension_worst    -3.6542      1.115     -3.276      0.001      -5.845      -1.463
co0pactness_0ean            3.0183      0.641      4.708      0.000       1.759       4.278
concave points_0ean        -3.6543      1.052     -3.474      0.001      -5.720      -1.588
concavity_worst            -0.4799      0.151     -3.176      0.002      -0.777      -0.183
concavity_se                1.8357      0.718      2.557      0.011       0.426       3.246
==============================================================================
Omnibus:                       24.593   Durbin-Watson:                   1.751
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               26.680
Skew:                          -0.524   Prob(JB):                     1.61e-06
Kurtosis:                       3.161   Cond. No.                     4.67e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.67e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
```

For backward selection, the predictors were: compactness_mean, concave_points_mean, radius_se, smoothness_se, concavity_se, concave_points_se, radius_worst, texture_worst, area_worst, concavity_worst, symmetry_worst, and fractal_dimension_worst. The eleven common variables identified are associated with concavity, compactness, radius, texture, area, and smoothness.

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              diagnosis   R-squared:                       0.769
Model:                            OLS   Adj. R-squared:                  0.764
Method:                 Least Squares   F-statistic:                     154.6
Date:                Wed, 13 Dec 2023   Prob (F-statistic):          3.04e-168
Time:                        04:22:49   Log-Likelihood:                 23.483
No. Observations:                 569   AIC:                            -20.97
Df Residuals:                     556   BIC:                             35.50
Df Model:                          12                                         
Covariance Type:            nonrobust                                         
===========================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------
const                       3.2010      0.174     18.443      0.000       2.860       3.542
co0pactness_0ean            3.4311      0.553      6.205      0.000       2.345       4.517
concave points_0ean        -4.3313      0.907     -4.777      0.000      -6.112      -2.550
radius_se                  -0.2445      0.068     -3.591      0.000      -0.378      -0.111
s0oothness_se             -19.6846      4.351     -4.524      0.000     -28.231     -11.139
concavity_se                3.3763      0.735      4.596      0.000       1.933       4.819
concave points_se          -8.6074      3.338     -2.578      0.010     -15.165      -2.050
radius_worst               -0.1329      0.014     -9.199      0.000      -0.161      -0.105
texture_worst              -0.0107      0.002     -5.889      0.000      -0.014      -0.007
area_worst                  0.0008      0.000      7.313      0.000       0.001       0.001
concavity_worst            -0.6867      0.144     -4.778      0.000      -0.969      -0.404
sy00etry_worst             -0.8979      0.207     -4.343      0.000      -1.304      -0.492
fractal_di0ension_worst    -4.0741      1.089     -3.740      0.000      -6.213      -1.935
==============================================================================
Omnibus:                       24.208   Durbin-Watson:                   1.764
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               26.210
Skew:                          -0.520   Prob(JB):                     2.03e-06
Kurtosis:                       3.161   Cond. No.                     4.71e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.71e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
```

However, variables such as perimeter and symmetry were not generally considered, possibly due to data multicollinearity (notably between perimeter and radius).
In the machine learning phase, the Decision Tree model exhibited a distinct preference for variables labeled as “worst,” with “perimeter_worst” emerging as the most critical feature. 
![Decision_Tree](https://github.com/Vivsquared/QTM-347-Final-Project/blob/641a4a1d91c3a9ef80327dd52eacdb8c8573d859/Machine%20Learning%20Models/Decision%20Tree.png)


This divergence in predictor selection between the forward/backward subset methods and Decision Trees could be attributed to the linear structure and multicollinearity sensitivity of the subset selection. In contrast, Decision Trees, as non-linear models, prioritize the efficacy of feature splitting, enabling them to discern more complex relationships and minimize the impact of multicollinearity. Hence, the perimeter, though often excluded in subset models due to high multicollinearity with the radius, is deemed vital in the Decision Tree model. The tuned Lasso model retained most predictors, excluding only four features with coefficients equals to 0, and still achieved a notably low mean squared error (0.054). The $\lambda$ for lasso model is around 7, indicating there is a penalty term impacting the value of cost function. 

![lasso](https://github.com/Vivsquared/QTM-347-Final-Project/blob/15eedc0c797f865f681a4045794d6e34641417e9/Machine%20Learning%20Models/lasso.png)

```
                Column Name  Coefficient
0               radius_mean    -0.000634
1              texture_mean    -0.009559
2            perimeter_mean    -0.018356
3                 area_mean    -0.016689
4           smoothness_mean     0.002235
5          compactness_mean     0.102501
6            concavity_mean    -0.047058
7       concave points_mean    -0.058170
8             symmetry_mean     0.000000
9    fractal_dimension_mean     0.014467
10                radius_se    -0.134616
11               texture_se     0.024322
12             perimeter_se    -0.002045
13                  area_se     0.093401
14            smoothness_se    -0.063909
15           compactness_se     0.032824
16             concavity_se     0.083571
17        concave points_se    -0.059457
18              symmetry_se     0.000000
19     fractal_dimension_se     0.001526
20             radius_worst    -0.431797
21            texture_worst    -0.077831
22          perimeter_worst     0.000000
23               area_worst     0.285011
24         smoothness_worst    -0.000000
25        compactness_worst     0.002278
26          concavity_worst    -0.063220
27     concave points_worst    -0.080741
28           symmetry_worst    -0.048880
29  fractal_dimension_worst    -0.056613
Optimal number of features: 26
```

This suggests a general correlation between diagnosis and all predictors, notwithstanding the detected multicollinearity.

 <br> Accuracy:
| Model         | Misclassification Rate   |
|---------------|--------------------------|
| Lasso Model   | 0.054                    |
| Decision Tree | 0.94                     |
| Clustering    | 0.090                    |

<br> 
Cross validation: 

<br>

### 4.2 Supplementary approaches
This study incorporated KMeans clustering to group similar data points based on all features. While clustering is not typically employed for accuracy determination in classification, it achieved a remarkably low mean squared error (MSE) of 0.0896. This result suggests a strong correlation between the features and the diagnosis outcome. The clustering exhibited clear separation with minimal overlap, indicating distinct groupings.
![KMeans_Clustering](https://github.com/Vivsquared/QTM-347-Final-Project/blob/1612a44815676a7fe4100e85896c63b22a463501/KMeans%20Clustering/clustering%20Image.png)
Consequently, the machine learning models employed were either based on the number of variables identified by subset selection or utilized all predictors. Each model slightly varied in the type of predictors used to enhance accuracy, but effective models consistently involved approximately 13 predictors or the entire set of predictors.

<br> Clustering MSE: 0.090

## 5. Discussion
Our study achieved high accuracy in tumor type prediction. Using the same dataset, a previous approach combined with an image-based dataset achieved a 75.52% accuracy rate without data filtering (Tan, 2020). In contrast, all methods in our study maintained accuracy rates above 90%. Both studies used 10-fold cross-validation, with the decision tree approach showing similarities. The accuracy discrepancy can be attributed to differences in predictor selection and the integration of imaging data in the Southern University of Science and Technology study. Tan's approach on accuracy based on the ratio of true and false positives, similar from our MSE-based calculation. Given that Tan’s coevolutionary neural network reached 88% accuracy before filtering the data, and considering the efficacy of resample filtering in enhancing accuracy in Tan’s study, these strategies could further improve our research.

## 6. Conclusion
Our study leverages Wisconsin’s open dataset on breast cancer, exploring a combination of clustering and classification methods alongside subset selection for predictor determination — an approach not previously undertaken. Focused on selecting effective variables for accurate tumor type prediction, the study signifies a milestone in breast cancer risk assessment and diagnostic speed enhancement. It involved two researchers: Researcher 1 handled subset selection and clustering, while Researcher 2 focused on the machine learning approach and cross-validation, and both researcher consistantly aims to enhance the accuracy of the prediction. The study's success in predicting tumor type underscores its potential impact on breast cancer diagnostics.

## 7. References
Bursac Z, Gauss CH, Williams DK, Hosmer DW. Purposeful selection of variables in logistic regression. Source Code Biol Med. 2008 Dec 16;3:17. doi: 10.1186/1751-0473-3-17. PMID: 19087314; PMCID: PMC2633005.

Mohammed SA, Darrab S, Noaman SA, Saake G. Analysis of Breast Cancer Detection Using Different Machine Learning Techniques. Data Mining and Big Data. 2020 Jul 11;1234:108–17. doi: 10.1007/978-981-15-7205-0_10. PMCID: PMC7351679.

Rabiei R, Ayyoubzadeh SM, Sohrabei S, Esmaeili M, Atashi A. Prediction of Breast Cancer using Machine Learning Approaches. J Biomed Phys Eng. 2022 Jun 1;12(3):297-308. doi: 10.31661/jbpe.v0i0.2109-1403. PMID: 35698545; PMCID: PMC9175124.

