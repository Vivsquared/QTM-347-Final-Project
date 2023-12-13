# Diagnostic Prediction - Breast Cancer 
**Authors: Jennifer Jiang, Vivienne Yu**

## Abstract
Breast cancer is the second most common cancer among women in the United States, but the diagnostic process can take weeks or months. Because of this situation, we would like to use machine learning models to reduce the diagnostic times. By testing different subset selection models, we have found 13 most effective variables,'concave points_worst', 'radius_worst', 'texture_worst', 'area_worst', 'smoothness_se', 'symmetry_worst', 'compactness_se', 'radius_se', 'fractal_dimension_worst', 'compactness_mean', 'concave points_mean', 'concavity_worst', 'concavity_se', to determine the dependent variable, 'diagonsis'. Then, we applied several machine learning models to see which one leads to the most accurate results and found that Logit Model and Random Forest produced the most accurate prediction for this classication diagnostic prediction project.

## Introduction

## 3. Result
### 3.1 Main Indications of the Result
This study comprises three distinct sections to determine breast tumor types and the presence of breast cancer: subset selection, machine learning methodology, and cross-validation.
Initially, the study employed two subset selection methods: forward selection and backward selection. Forward selection is typically used when the number of predictors exceeds the number of samples, whereas backward selection is preferred when the sample size surpasses the number of predictors. Given our dataset of approximately 500 samples and 30 predictors, backward selection was initially applied. However, due to the relatively large number of predictors and limited sample size, and considering our extensive set of covariates, forward selection was deemed more beneficial (Bursac, 2008). The effectiveness of predictors in subset selection was evaluated based on R-squared values and P-values, with a higher R-squared and a P-value under 0.05 as the criteria for effective predictor selection. Consequently, the predictors chosen for forward selection included: concave_points_worst, radius_worst, texture_worst, area_worst, smoothness_se, symmetry_worst, compactness_se, radius_se, fractal_dimension_worst, compactness_mean, concave_points_mean, concavity_worst, and concavity_se. For backward selection, the predictors were: compactness_mean, concave_points_mean, radius_se, smoothness_se, concavity_se, concave_points_se, radius_worst, texture_worst, area_worst, concavity_worst, symmetry_worst, and fractal_dimension_worst. The eleven common variables identified are associated with concavity, compactness, radius, texture, area, and smoothness. However, variables such as perimeter and symmetry were not generally considered, possibly due to data multicollinearity (notably between perimeter and radius).
In the machine learning phase, the Decision Tree model exhibited a distinct preference for variables labeled as “worst,” with “perimeter_worst” emerging as the most critical feature. This divergence in predictor selection between the forward/backward subset methods and Decision Trees could be attributed to the linear structure and multicollinearity sensitivity of the subset selection. In contrast, Decision Trees, as non-linear models, prioritize the efficacy of feature splitting, enabling them to discern more complex relationships and minimize the impact of multicollinearity. Hence, the perimeter, though often excluded in subset models due to high multicollinearity with the radius, is deemed vital in the Decision Tree model. The tuned Lasso model retained most predictors, excluding only four features, and still achieved a notably low mean squared error (0.054). This suggests a general correlation between diagnosis and all predictors, notwithstanding the detected multicollinearity.
Cross validation: 

### 3.2 Supplementary approaches
This study incorporated KMeans clustering to group similar data points based on all features. While clustering is not typically employed for accuracy determination in classification, it achieved a remarkably low mean squared error (MSE) of 0.0896. This result suggests a strong correlation between the features and the diagnosis outcome. The clustering exhibited clear separation with minimal overlap, indicating distinct groupings.
![image](https://github.com/Vivsquared/QTM-347-Final-Project/blob/1612a44815676a7fe4100e85896c63b22a463501/KMeans%20Clustering/clustering%20Image.png)
Consequently, the machine learning models employed were either based on the number of variables identified by subset selection or utilized all predictors. Each model slightly varied in the type of predictors used to enhance accuracy, but effective models consistently involved approximately 13 predictors or the entire set of predictors.

## 4. Discussion
Our study achieved high accuracy in tumor type prediction. Using the same dataset, a previous approach combined with an image-based dataset achieved a 75.52% accuracy rate without data filtering (Tan, 2020). In contrast, all methods in our study maintained accuracy rates above 90%. Both studies used 10-fold cross-validation, with the decision tree approach showing similarities. The accuracy discrepancy can be attributed to differences in predictor selection and the integration of imaging data in the Southern University of Science and Technology study. Tan's approach on accuracy based on the ratio of true and false positives, differing from our MSE-based calculation. Given that Tan’s coevolutionary neural network reached 88% accuracy before filtering the data, and considering the efficacy of resample filtering in enhancing accuracy in Tan’s study, these strategies could further improve our research.

## 5. Conclusion
Our study leverages Wisconsin’s open dataset on breast cancer, exploring a combination of clustering and classification methods alongside subset selection for predictor determination — an approach not previously undertaken. Focused on selecting effective variables for accurate tumor type prediction, the study signifies a milestone in breast cancer risk assessment and diagnostic speed enhancement. It involved two researchers: Researcher 1 handled subset selection and clustering, while Researcher 2 focused on the machine learning approach and cross-validation, and both researcher consistantly aims to enhance the accuracy of the prediction. The study's success in predicting tumor type underscores its potential impact on breast cancer diagnostics.

## 6. References
Bursac Z, Gauss CH, Williams DK, Hosmer DW. Purposeful selection of variables in logistic regression. Source Code Biol Med. 2008 Dec 16;3:17. doi: 10.1186/1751-0473-3-17. PMID: 19087314; PMCID: PMC2633005.

Mohammed SA, Darrab S, Noaman SA, Saake G. Analysis of Breast Cancer Detection Using Different Machine Learning Techniques. Data Mining and Big Data. 2020 Jul 11;1234:108–17. doi: 10.1007/978-981-15-7205-0_10. PMCID: PMC7351679.

