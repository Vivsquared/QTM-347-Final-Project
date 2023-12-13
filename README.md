# Diagnostic Prediction - Breast Cancer 
**Authors: Jennifer Jiang, Vivienne Yu**

## Abstract
Breast cancer is the second most common cancer among women in the United States, but the diagnostic process can take weeks or months. Because of this situation, we would like to use machine learning models to reduce the diagnostic times. By testing different subset selection models, we have found 13 most effective variables,'concave points_worst', 'radius_worst', 'texture_worst', 'area_worst', 'smoothness_se', 'symmetry_worst', 'compactness_se', 'radius_se', 'fractal_dimension_worst', 'compactness_mean', 'concave points_mean', 'concavity_worst', 'concavity_se', to determine the dependent variable, 'diagonsis'. Then, we applied several machine learning models to see which one leads to the most accurate results and found that Logit Model and Random Forest produced the most accurate prediction for this classication diagnostic prediction project.

## Introduction

## 3. Result
### 3.1 Main Indications of the Result
This study comprises three distinct sections to determine breast tumor types and the presence of breast cancer: subset selection, machine learning methodology, and cross-validation.

<br> The study employed two subset selection methods: forward selection and backward selection. Forward selection is typically used when the number of predictors exceeds the number of samples, whereas backward selection is preferred when the sample size surpasses the number of predictors. Given our dataset of approximately 500 samples and 30 predictors, backward selection was initially applied. However, due to the relatively large number of predictors and limited sample size, and considering our extensive set of covariates, forward selection was deemed more beneficial (Bursac, 2008). The effectiveness of predictors in subset selection was evaluated based on R-squared values and P-values, with a higher R-squared and a P-value under 0.05 as the criteria for effective predictor selection. Consequently, the predictors chosen for forward selection included: concave_points_worst, radius_worst, texture_worst, area_worst, smoothness_se, symmetry_worst, compactness_se, radius_se, fractal_dimension_worst, compactness_mean, concave_points_mean, concavity_worst, and concavity_se.
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


This divergence in predictor selection between the forward/backward subset methods and Decision Trees could be attributed to the linear structure and multicollinearity sensitivity of the subset selection. In contrast, Decision Trees, as non-linear models, prioritize the efficacy of feature splitting, enabling them to discern more complex relationships and minimize the impact of multicollinearity. Hence, the perimeter, though often excluded in subset models due to high multicollinearity with the radius, is deemed vital in the Decision Tree model. The tuned Lasso model retained most predictors, excluding only four features with coefficient equals to 0, and still achieved a notably low mean squared error (0.054).
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
<br> lasso model MSE: 0.054
<br> Decision Tree percent accuracy: 0.94

<br> 
Cross validation: 

### 3.2 Supplementary approaches
This study incorporated KMeans clustering to group similar data points based on all features. While clustering is not typically employed for accuracy determination in classification, it achieved a remarkably low mean squared error (MSE) of 0.0896. This result suggests a strong correlation between the features and the diagnosis outcome. The clustering exhibited clear separation with minimal overlap, indicating distinct groupings.
![KMeans_Clustering](https://github.com/Vivsquared/QTM-347-Final-Project/blob/1612a44815676a7fe4100e85896c63b22a463501/KMeans%20Clustering/clustering%20Image.png)
Consequently, the machine learning models employed were either based on the number of variables identified by subset selection or utilized all predictors. Each model slightly varied in the type of predictors used to enhance accuracy, but effective models consistently involved approximately 13 predictors or the entire set of predictors.

<br> Clustering MSE: 0.090

## 4. Discussion
Our study achieved high accuracy in tumor type prediction. Using the same dataset, a previous approach combined with an image-based dataset achieved a 75.52% accuracy rate without data filtering (Tan, 2020). In contrast, all methods in our study maintained accuracy rates above 90%. Both studies used 10-fold cross-validation, with the decision tree approach showing similarities. The accuracy discrepancy can be attributed to differences in predictor selection and the integration of imaging data in the Southern University of Science and Technology study. Tan's approach on accuracy based on the ratio of true and false positives, differing from our MSE-based calculation. Given that Tan’s coevolutionary neural network reached 88% accuracy before filtering the data, and considering the efficacy of resample filtering in enhancing accuracy in Tan’s study, these strategies could further improve our research.

## 5. Conclusion
Our study leverages Wisconsin’s open dataset on breast cancer, exploring a combination of clustering and classification methods alongside subset selection for predictor determination — an approach not previously undertaken. Focused on selecting effective variables for accurate tumor type prediction, the study signifies a milestone in breast cancer risk assessment and diagnostic speed enhancement. It involved two researchers: Researcher 1 handled subset selection and clustering, while Researcher 2 focused on the machine learning approach and cross-validation, and both researcher consistantly aims to enhance the accuracy of the prediction. The study's success in predicting tumor type underscores its potential impact on breast cancer diagnostics.

## 6. References
Bursac Z, Gauss CH, Williams DK, Hosmer DW. Purposeful selection of variables in logistic regression. Source Code Biol Med. 2008 Dec 16;3:17. doi: 10.1186/1751-0473-3-17. PMID: 19087314; PMCID: PMC2633005.

Mohammed SA, Darrab S, Noaman SA, Saake G. Analysis of Breast Cancer Detection Using Different Machine Learning Techniques. Data Mining and Big Data. 2020 Jul 11;1234:108–17. doi: 10.1007/978-981-15-7205-0_10. PMCID: PMC7351679.

