# SemesterProject
This is the semester project for EECS 731 data science.

# team member:
Kun He, 

Chiehen Hung,  

Ben Liu, 

Guojun Xiong,  

Tianxiao Zhang,  

Xiaohan Zhang

# structure

-- folder that contains all data files --

/data

/data/Beijing_housing_price.csv

/data/kc_house_data.csv

-- notebook for anomaly detection --

/Anomaly_detection_Beijing.ipynb

/Anomaly_detection_KC.ipynb

-- notebook for regression and time series forecasting --

/Housing_predict-Beijing.ipynb

/Housing_predict-KC.ipynb

-- notebook for clustering --

cluster.ipynb

-- notebook for classification --

/Classification_Beijing.ipynb

/Classification_kc.ipynb

# what we did
## Anomaly Detection

For anomaly detection, there are three main anomalies: Point Anomalies, Contextual Anomalies, Collective Anomalies.

In this project, we will mainly process two dataset: Beijing housing Price and KC Housing price. For these two dataset, we will utilize some anomaly detection approaches to find the outliers of the dataset. The main algorithms we utilize for anomaly detection are Interquartile Range (IQR), Isolation Forest, One Class SVM, Local Outlier Factor, Robust Covariance and K-Nearest Neighbors (KNN). Finally, we will combine these models together to calculate the final anomaly score for each data and select the anomalies which satisfy some threshold of the score, which is the safest way to do unsupervised anomaly detection.

Since this is a problem of unsupervised anomaly detection, we will judge the performance of each method by visualizing the anomalies among the entire dataset. We think this is an efficient way to make the judgement of the results.

## Time Series Forecasting

Thia part aims at predicting the trend of housing price for Beijing and KC. We transform the data sets into the time series data frame and characterize the trend of housing price for the given data sets.  The housing price in Beijing has the potential to grow in the near future, while the trend of that of KC will keep stationary for a long period.

Two models are adopted for the time series forecasting: 

Model 1: ARIMA
Model 2: Prophet 


For Beijing housing price data, both models can predict the increasing trend in the future, while the Prophet can outperform ARIMA in the sense of actual housing price.

For KC housing price data, we only demenstrate the trend predicted by Prophet, since the trend keeps stationary  in accordance with the original data. 

## Regression

This part aims at predict the exact housing price with particular feathers. We adopt the following three models:

Model 1: Linear Regression
Model 2: Linear SVM
Model 3: Neural Network

The original data sets were divided into training and testing data set with the ratio of 80% and 20%. All the regressors demenstrate high accuracy for prediction and among them, NN is the best due to nonlinear property.

## Clustering

The original purpose of clustering are (1) to help regression by divide the data set into different clusters, and then train individual model for each cluster  and (2) to showcase that by considering very little information, the clusters can predict the housing price. 

Using the geographic information, the result shows that,  K-means clusters the data into clusters that are fairly close to the actual subregions of each city, using K=13 for Beijing, and K=19 for KC, where 13 is the number of adminstrative districts in Beijing and 19 is the number of counties in KC. Similar result is achieved by Birch using the default set up. 
The average housing price of each cluster varies and they are different from each other even for those that are right next to each other. This result indicates that we should be able to distinguish the data point by using more information and have a decent classification result.

## Classification

### Beijing Housing Price Classification:
1. Two kinds of labels are generated based on unit price
	* 1.1 Equally spaced 
	* 1.2 Equally area   
  
#### Purpose
The purpose of doing is to see how labels could influence the result. As we know, for labels are given by people’s will, for example, movie rating and etc. So i got the freedom to choose the way that i would like to label the data. While plotting all the data with many bins, new found it actually follows Gaussian distribution, in this way, the label generated should also follow gaussian distribution closely, which turns out it does! 

#### Result
And the second method of labeling is just an extreme experiment, that i want to create a method of labeling that has the exact same number of samples for each class. After i plotted the price map based on equally area label, it turns out that it wasn’t a good way of labeling. 
Staring here, only equally spaced labeling method was applied to do the classification. Six models were applied:

   * 1. Gradient Boosting Classifier: 46.1%
   * 2. Random Forest Classifier:     43.8%
   * 3. Neural Network:               43.2%
   * 4. Decision Tree Classifier:     38.3%
   * 5. KNeighbors:                   36.6%
   * 6. SVM:                          29.2%
   
#### Result with 1 class tolerance
Consider that the dataset is time related. As we known from time series, timing has a huge influence in unit price as well. Base on this, think of some close edge situations, we give it a 1 class tolerance and would like to see how that works. Here goes the result:

   * 1. Neural Network:               87.3%
   * 2. Gradient Boosting Classifier: 87.0%
   * 3. Random Forest Classifier:     84.8%
   * 4. Decision Tree Classifier:     80.0%
   * 5. KNeighbors:                   74.3%
   * 6. SVM:                          68.2%

### King County(KC) Housing Price Classfication:

Step1: Feature Engineering
        1)
        square = sqft_living + sqft_lot + sqft_above + sqft_basement
        which means we plus all the square categories to get the total square of the house

        2)
        UnitPricee = price/square
        To generate a common feature for house price

Step2: Define the label(feature) we want 
        Here basically I divide the UnitPrice in same range with each range is 10 dollars/sqft 
        The first ten label are 0-10, 10-20, ..., 90-100.
        Then I generate the following labels with 100-120, 120-150, 150-infinity.
        The reason is that data point which is larger than 100 actually stands a little part of the data set, 
            divide data in such way can make data distribute like a perfect Gaussian distribution.

Step3: Classification
        Here I use 6 six different clssifier to make classification. These are the classifiers and the result:

        1.Random Forest:         54.5%
        2.Gradient Boosting      52.7%
        3.Decision Tree          46.0%
        4.KNN                    28.0%
        5.SVM                    20.7%
        6.Neural Network         42.5%

        And I choose RF, GB, NN for the final choice.

        Consider that sometime the different between exact unit price and predict unit price actually is small, but it will be at different label, such as 19.9 and 20.1
        So I give it a tolerance with 1 possible class shift, which means class 1/2/3 = class 2.

        With this operation, the result is imporved:
        1.RF                    90.3%
        2.GB                    88.9%
        3.NN                    81.7%
