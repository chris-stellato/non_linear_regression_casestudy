# Regression Case Study

#### (Ben) 
## The Challenge: 
Our goal is to build a machine learning model that makes a prediction for sale price of a piece of heavy equipment at auction based on its equipment type, usage, and configuration. 

## The Data: 
A dataset containing the results of over 400k auctions of heavy farm equipment

## Our approach: 

- EDA 
- Select columns we could immediately throw out

#### (Chris) 
- Create multiple helper functions to help us clean the data:
  - Transform text-based columns with less than 10 unique values into dummy columns
  - Use dummy columns to deal with nans
  - With our numerical column (machine hours), replace all nans with median value
  - scale the data
  - create a train test split
- create a helper function that would fit various models to our data when excuted
- create an aggregator function to run all of the helper functions
- ran a grid search to find optimized hyperparameters (run-time was too long, computer did not finish prior to deadline) 
- measure and plot feature permutation importance to help us understand which features influence predictions the most
- validate against initial holdout data

#### (Joseph)
## Our results: 
- Our GradientBoostedRegressor yeilded R2 score of 0.519 with defualt parameters. 
- Adjusting defaults, we found a lower learning rate (0.05), higher n_estimators, and higher max_depth increased our score to 0.539
  - due to timing out on the grid search, we manually adjusted hyperparameters to improve our score. 
- Using a DecisionTreeClassifier, we were able to obtain a score of 0.471 against our holdout data. 


![Lower_rate](https://user-images.githubusercontent.com/70020774/111010884-04bebd00-8355-11eb-9bf3-96b04c9dd4c3.png)

![bestscore](https://user-images.githubusercontent.com/70020774/111011325-3edc8e80-8356-11eb-8120-eddba379cda2.png)


#### (Austin)
## Our challenges: 
- We started out feeling confident about our plan and our project, but as day progressed it started to get away from us... 
- We significantly underestimated two key pieces of the project workflow: 
  1. We underestimated how much time data wrangling would take, and getting our data prepared for our model
  2. We underestimated the additional complexities of building a .py file with helper functions, and how much harder that is to troubleshoot than using a notebook. (takes longer to troubleshoot on the frontend, but .py files are more easily used in a production environment). 
- Due to time limitations, we were not able to test our model against the initial holdout data. 
- We were not confident in the outcome of our feature permutations importance graph

![uhoh](https://user-images.githubusercontent.com/70020774/111011485-d7730e80-8356-11eb-9ff1-eb50edcaf54c.png)



## What would we do differently? 
- Begin writing OOP style from the beginning, giving us more time to troubleshoot helper functions
- Take a different approach to dealing with non-numerical and nan data earlier in the process (we lost a significant amount of time going down several wrong paths) 
