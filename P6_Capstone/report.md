# Machine Learning Capstone Project
## 1. Introduction

Financial problem is one the serious problems that may drag people from heaven back to earth. For example, some people may quit college because of the tuition or supporting the family. We may realize that a term deposit which is like a self-funded insurance and a back-up plan may help avoid or reduce the influence of them.

We would like to build classifiers based on supervised learning algorithms to make classification about if a client has subscribed a term deposit. We also would like to analyse the relationship of different features, and if the observations can be clustered into several groups.

The dataset is obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and contains 45211 observations and 17 variables (including response and predictors).

## 2. Data Exploration
### 2.1 Response

The response variable `y` has two values "yes" and "no" here, which stands for whether the client has subscribed a term deposit or not, respectively. We would like to use the frequency table below to show the general distribution of our response variable.

|      y     |  yes  |   no  | total |
|:----------:|:-----:|:-----:|:-----:|
|  Freqency  |  5289 | 39922 | 45211 |
| Percentage | 11.7% | 88.3% |  100% |

Obviously, our response variable `y` here does not follow a balanced distribution. Because around 88.3% of the total observations have value "no" while only the rest 11.7% have value "yes". Luckily, there is no missing values here.

### 2.2 Removed Variables

There are two variables `day` and `month` in our dataset which stand for last contact day of month and last contact month of year, respectively. I think that these two variables should be considered as categorical variables even though variable `day` were recorded by using numerical values. Because the values of these two variables do not actually have the corresponding quantitative meanings here. Instead, these two are kind of the time index of the last contact time.

Hence, I would like not to take these two variables into consideration for the rest of this project.

### 2.3 Continuous Variables

There are 6 continous variables in our dataset. They are "age", "balance", "duration", "campaign", "pdays" and "previous". The description and the scatter matrix plot with diagonal being density plot of them are shown below.

- age: how old the client is?
- balance: the current balance in the bank
- duration: last contact duration, in seconds (numeric). __Important note__: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)

![scatter](https://github.com/whuang67/Udacity_Machine_Learning/blob/master/P6_Capstone/scatter.png?raw=true)

From the description and plot above, I have several observations here.

__1.__ Variable `duration` is highly correlated with our response variable `y`. If `duration` is 0, then `y` has to be "no".  
__2.__ The value 999 of variable `pdays` does not actually stands for 999 days that passed by after the client was last contacted from a previous compaign. What it actually indicates is missing values and that this client has never been contacted before. But actually there is no point with 999 `pdays` value.  
__3.__ There is one observation with extremely high `previous` value. We would like to remove that point to lower the influence that may be brought from that observation. The new scatter plot is shown below, and it looks much better.

![scatter2](https://github.com/whuang67/Udacity_Machine_Learning/blob/master/P6_Capstone/scatter2.png?raw=true)

Since we remove only 1 observation, the distribution of our response variable `y` will not change dramatically. It will still be an unbalanced distribution.

### 2.4 Categorical Variables

The rest 8 variables in our dataset are all categorical variables. They are "job", "marital", "education", "default", "housing", "loan", "contact" and "poutcome". The description of them are shown below.

- job: type of job ("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "selfemployed", "services", "student", "technician", "unemployed", "unknown")
- marital: marital status ("divorced", "married", "single", "unknown"; note: "divorced" means divorced or widowed)
- education: education level ("secondary", "tertiary", "primary", "unknown")
- default: has credit in default? ("no", "yes")
- housing: has housing loan? ("no", "yes")
- loan: has personal loan? ("no", "yes")
- contact: contact communication type ("cellular", "telephone", "unknown")
- poutcome: outcome of the previous marketing campaign ("failure", "other", "success", "unknown")
