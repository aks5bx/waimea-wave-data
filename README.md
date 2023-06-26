# **World Surf League Competition Analysis**

# Overview 

## General Information

Author: Adi Srikanth 

Date: June 2023 

This repository contains code for the World Surf League project intended for Strong Analytics. The project description and the data are not included in this repository in order to protect the confidentiality of the project as stipulated by Strong Analytics. 

This repository is meant to be a starting point for project completion. There are many parts of the data science process that are incomplete as a result of time constraints. Limitations are acknowledged throughout the writeup. 

## Disclosure 

This project was completed with assistance from GitHub CoPilot, used via an extension to VSCode. Additionally, this project utilized Google Colab for running a model prototype. 

No other assistance was used for this project. The total time spent on this project was around four hours. 

## Repository Overview 

The repository can be cloned over ssh as supported by GitHub. Additionally, the required packages (with correct versions) can be installed using the command `pip install -r requirements.txt`. 

```
[parent or home directory]
│
└───data
│   | buoy-data.csv
│   | wide.csv
│
└───prototyping
│   | xgboost.ipynb
│   | mean_shap.png
│   | shap_summary.png
│
└───analysis
│   | wsl_analysis.pdf
│   | project_overview.pdf
│
└───.gitignore
└───README.md
└───requirements.txt
```

# Data

## Overview 

As noted, the data was provided by Strong Analytics and is not included in this repository. However, the data is in csv format and includes mostly numeric values. Some data previews and plots can be viewed from the jupyter notebook `xgboost.ipynb`.

## Exploratory Data Analysis 

Exploratory data analysis can be found in the notebook `xgboost.ipynb`. The two main takeaways are the existence of missing data and the skewed distribution of the target variable. 

The data includes various buoy readings, taken and reported on a daily basis. There are five buoys in total, including one near Tokyo, one near Alaska, and three near Hawaii. Of the three buoys near Hawaii, one is specifically tagged as the buoy by Waimea Bay and is therefore the buoy that we use as our best approximation of the wave conditions at Waimea Bay. 

## Missing Data 

There is significant missing data in the dataset provided. Generally, missing data is seen on a buoy by buoy basis. As in, if a buoy is down (and its data is missing), all of the metrics it was supposed to generate are missing. If a buoy is active, it generally provides all the data it should provide. 

In order to handle the missing data, we analyze possible trends that underpin the missing data. For example, if buoy data tend to be missing during a specific set of environmental/wave conditions, missing data might actually indicate valuable information. If there is not a clear trend however, missing data can be treated as less important or indicative. After our analysis, we find that the Tokyo and Alaska buoys do not exhibit an evident trend with their missing data. The distributions of our target variable and our other features are largely the same regardless of whether the Tokyo/Alaska buoy data is missing or not. However, the Hawaii buoys _do_ exhibit trends with respect to missing data. 

### Tokyo and Alaska Buoys 

Because we did not find a clear trend amongst the missing data for these buoys, we choose to address the missing data here by imputing missing values with the median value of the feature. This allows us to retain rows with missing data while avoiding prescribing significance to missing data. 

### Hawaii Buoy Features

The Hawaii buoys did appear to exhibit missing data with some trend pertaining to environmental factors. So, instead of imputing missing values, we choose to represent missing values as their own categorical class. This allows any future model to learn potential relationships between the lack of data and our target variable. 

In order to do this, we take our numeric buoy data and convert them into categorical classes. Specifically, we take each feature and convert the numeric values to their respective quantiles and treat each quantile as a category. Missing data points then become their own, sixth category. 

_Note: the xgboost package used for rapid prototyping does not support categorical variables. So, for a first attempt, we allow the model to treat these are numeric. Other distributions of xgboost allow for categorical variables._

# Feature Engineering

## Target Variable 

To simplify our task for a first attempt, we convert our numeric target variable into a categorical variable. We know that wave height must be at least three meters to support a successful surf competition. So, we take our wave height target variable and represent it as a binary variable indicating whether or not the wave height is at least three meters. There is significant class imbalance in this variable, which informs the model choice for this dataset. 

## Date

We extract the year and month from our date feature. Like the Hawaii buoy features, we treat these months and years as categorical variables. (Given nonlinear seasonality, we do not want to impose a continuous relationship between date variables and our target variable). 

_Note: the xgboost package treats these as numeric variables. See the note under Hawaii Buoy Features_

# Modeling 

## Overview 

In order to generate a starting point for this project, we implement a model that can give us an understanding of the relationship between _current conditions_ and _current wave height_. Of course, we ultimately need to be able to use current conditions to predict _future_ wave height. However, we leave that for future implementation given its additional complexity. 

We use an xgboost model as its tree-based structure makes it robust to outliers and to the imbalanced target variable class. It also is well-designed to pick up on nonlinear relationships and interactions between categorical and continuous data. 

## Model Implementation 

The model is implemented using the scikit-learn out-of-the-box xgboost module. This module is severely limited as it allows for minimal customization and notably does not support categorical variables at this time. Future development should be done using xgboost implementations available in Spark (or from scratch in PyTorch). 

We evaluate the model using precision instead of accuracy. Given our target variable class imbalance, a high accuracy can be achieved by blindly predicting a positive value of our target variable. Instead, we want to make sure that when we claim that our target variable is positive (i.e. that the wave conditions can support a competition), our claim is actually true. So, we use precision in our model evaluation. 

## Results 

Using a three-fold cross validation, we get a precision score of, on average, 70%. 

This is a reasonable starting point, especially given the fact that we did not conduct hyperparameter tuning, our model implementation did not support the proper variable classes, and that the model itself was relatively simple with no added data to our dataset. 

## Analysis 

In order to understand the underlying relationships between current environmental conditions and current wave height, we leverage SHAP values to dive under the hood and determine which features impact our target variable and how they do so. SHAP value plots can be found in the notebook `xgboost.ipynb`. 

Understanding the relationships between current environmental conditions and current wave height can help us prioritize features when we ultimately predict future wave height. 

# **Next Steps**

The future work can be a writeup of the length of this repository's writeup in and of itself. However, we briefly summarize some next steps for this project. 

## Data 

It is crucial to collect more data for this project. The dataset itself is fairly small, which creates a high amount of variance in output, evidence by the variance in precision metrics computed across three folds. Additionally, more data points are likely necessary given the low initial precision (which is for current wave height, a theoretically simpler problem) and the number of features with a low SHAP value magnitude. 

## Modeling 

As mentioned, we are using current data to predict current conditions in the model above. We want to use current data to predict future conditions, a much more complex problem. 

In order to fit this problem, we should look to utilize a model framework that is better suited to take into account data that has a temporal dimension. An example of this would be a Recurrent Neural Network, which takes in data from multiple points in time in order to make a prediction. Versions of RNN also learn to place different levels of importance on different inputs to the RNN. In this use case, we could supply not only current environmental conditions, but also environmental conditions from the previous days and weeks. Then, our model could use all of this data to extrapolate a trend to predict forwards. 

## Analysis 

It is important to note that while predicting wave conditions is integral to this project, our end goal is to help the WSL host a successful surf competition. As such, we need to transform our wave condition predictions into an actionable insight for the WSL. 

To do this, we should use our predictions to find the time periods that are best to host a competition. We can do this by running predictions for each day in a 6-month time frame. Then, we can extract the prediction probabilities from our model for each prediction and rank them from highest to lowest. This allows us to get a ranked list of dates based on likelihood of favorable conditions. 

## Next Month Timeline 

If we had an additional month to work on this project, this is a sample timeline that could follow. 

### Week 1: Data 

In our first week, our priority would be to acquire additional data that is valuable to our end goal. This process would include research, expert interviewing, and data engineering efforts. 

### Week 2: Data & Rapid Prototyping 

In our second week, we would look to wrap up data collection and institute reliable data pipelines to ingest data from our data sources. These would ideally be automated and perhaps attached as a severless job through some cloud provider. 

Beyond data, we should begin rapid prototyping of models in order to get a sense for what type of model would be successful for us. 

### Week 3: Model Implementation 

In our third week, we could look to finalize our model and deploy it for repeated use. This could look like generating an API endpoint attached to some a compute resource (likely through a cloud provider) that we could hit in order to run the model on new data. 

Model implementation should also include hyperparameter tuning and other performance tuning to get optimal precision. 

### Week 4: Reporting and Automation 

In our final week, we should look to wrap up our project. Our model results should be converted into interpretable, business-ready insights for the WSL. Additionally, we should attempt to automate any data engineering and model infrastructure that we can in order to allow WSL to use our solution moving forward. 

### Caveats 

This is a highly idealistic timeline for the next month. In reality, we expect curveballs and forks in the road. It is important to be resilient and adaptable as this process unfolds. We also should continue to interface with WSL as much as possible as we iterate on solutions. 