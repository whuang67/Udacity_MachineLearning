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

### 2.2 Continuous Variables

There are 6 continous variables in our dataset. They are "age", "balance", "duration", "campaign", "pdays" and "previous".

- age: how old the client is?
- balance: the current balance in the bank
- duration: last contact duration, in seconds (numeric). __Important note__: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)

