# Analysis of Department of Commerce and Department of Labor Data
In this analysis, I have prepared a linear regression machine learning model, as well as attempting to prepare a neural network machine learning model. The goal of this analysis is to create a model that can predict agency turnover.  I used datasets from the Department of Commerce to create and train my models and tested the models against fresh data from the Deparmtnet of Labor. There are roughly 20,000 records from the Deparment of Commerce and 6433 records from the Department of Labor. 

## Data Exploration
I created a graph that compared all the value counts for the features we were focusing on. This shows us a good snapshot of how the data is skewing and an idea of what kind of data is missing. Question 84 has 2 answers that are non-numeric so that is showing an abnormally large amount of missing data that will later be recoded. 
![value_counts_overall_barh](https://github.com/wanderfarther/machine-learning-project/assets/132155105/112767f7-01da-42d8-a7f2-4f458ff48279)

I also graphed the value counts of specific features to visualize their distribution. I visualized how many employees reported military service. I did not beleive it was significant enough to delete any data. I visualized the division of male vs female employees and how the outcomes were effected. There is a relatively even distribution, with 10,065 male and 9,091 female employees. After analysing the outcomes of each group, each has about 25% of the employees reporting they would be leaving in the next year. The overall outcome graph showed similar data, with 26% of overall employees reporting they woiuld be leaving.

## Data Cleaning
### Redcuding the Data
I reduced the data from 116 columns to 31 columns initially. I reduce and combine some columns later.

### Removing unnecessary Data
In this section, I removed any rows that did not have a response to DLEAVING. This survey is aimed at predicting whether or not an employee is planning on leaving the compnay. If the survey taker did not indicate this, their data is not helpful to this study. I also removed data rows that have more the 15% of the data in the row missing. This removed noise that would hinder the models later on.

### Standardizing and recoding the data
In this section I standardized the data. After removing rows with more than 15% missing data, there were still rows with missing data. I replaced that missing data with an average of the feature for each missing data peice. I did this before I recoded question 84 and created dummy variavbles in order to not skew the average. I chose to replace both X and Y in question 84 to a 3 for simplicity. I also made sure that all my dtypes were correct for the next step of creating dummy variables.

### Creating Dummy Variables
In this section, I created dummy variables for all the demographic data. Before doing this though, I manually recoded the outcome variable to create a binary outcome. I also dropped 1 of each of the dummy columns after they were created to reduce the noise in the model.

### Creating a Correlation Matrix
![commerce_correlation_before](https://github.com/wanderfarther/machine-learning-project/assets/132155105/0c6aa48f-73f2-4550-9035-2111550b773a)
The above matrix is the initial matric of the features

In this section, I created a correlation matrix to see if any features highly corrlated with others. I chose to combine questions 48, 49, 50, and 52. All of these features highly correlate to one another and would have added extra, unnecessary noise in the models. I also chose to combine questions 85, 86, and 87. The first two questions are very similar, the first asking about being inspired by the job and the next is a sense of accomplishment. These are very similar concepts and the questions have a correlation of 82%. The third question also has a high coorelation with the first two, being between 70 and 74 percent. The last columns I combined were 55, 56, and 60. These questions were all about the upper managment and had correlations around 78%.

![commerce_correlation_after](https://github.com/wanderfarther/machine-learning-project/assets/132155105/77874d1e-aa56-437f-86d7-5c7234b7bc5a)
This is the correlation matrix after the highly correlated features have been combined.


## Logistic Regression Model
In  this section, I split my data between the outcome variable and the features, before then splitting it further into training and testing data. I intitally used the 'lbfgs' solver with a penalty of 'None'. I had to increase my max_iter value to 175 because the model 'failed to converge' in fewer iterations.
![logistic_regression_1](https://github.com/wanderfarther/machine-learning-project/assets/132155105/cddd3ce3-2460-4d3e-9aaa-41292280bae5)
![logistic_regression_1_cm](https://github.com/wanderfarther/machine-learning-project/assets/132155105/359e5a88-d3f2-4572-a8a5-9bbb1e2dc4ed)
![logistic_regression_1_predictions](https://github.com/wanderfarther/machine-learning-project/assets/132155105/80fc69af-e77d-4b60-967e-0f172cb25eed)

The model has a balanced accuracy rate of 62.8%. The prediction value for an employee not leaving the company is .86, while the prediction value for an employee leaving the company is .42. This means that the model is better at predicting an employee will stay with the company than if the employee will leave.

### SHAP Analysis
I used the python package SHAP to visualize which features were having the most impact on the outcomes. The summary plot was the most helpful. 
![SHAP_summary_plot](https://github.com/wanderfarther/machine-learning-project/assets/132155105/9aa22844-bdfc-46bb-912b-b9ed3ce3f535)

It shows that the top 2 factors affecting the outcome predictions were P_INSPIRED, which is an average of questions 85, 86, and 87. These questions are all personal reflection questions about the employee's connection to the organization and the sense of accomplishment they feel at work. 6_P_TAL_USE is the second feature, which is asks how well an employees talents are utilized at work. Most of the answers to the questions skew to the negative side of the graph, meaning most employees are satisifed with their connection to their job, but other side of the top 2 features have long blue lines. The first feature P_INSPIRE has an almost .8 SHAP value, while the second has closer to .5 or .6. This means that as an employee feels underutilized and less accomplioshed at work, the more likely they are to find other employement.

The next 2 most important features are the features that averaged a few upper-mangement questions and supervisory questions. These questions ask about management and their ability to motivate employees, their integrity and communication. As employees responded with lower scores to these questions, the more likely they are to seek other employement.

### Logistic Regression Model Optimization
Around 26% of the data was of employees who indicated they would be leaving with in the next year. This means the model has almost triple the data for an employee who is staying. This could explain the skewing towards employees staying in their postion. I used RandomOverSampler to create a more balanced amount of negative and positive data in the training data. I also changed the solver to 'sag', Stochastic Average Gradient descent, becaues it is also sutable for binary model outputs.
![logistic_regression_2](https://github.com/wanderfarther/machine-learning-project/assets/132155105/5c163150-9307-4fb0-bafa-d24cf4fc9321)
![logistic_regression_2_cm](https://github.com/wanderfarther/machine-learning-project/assets/132155105/cc3bb35a-f21d-4c8e-8766-749aa97dbb30)
![logistic_regression_2_precistions](https://github.com/wanderfarther/machine-learning-project/assets/132155105/2ca27826-7b3d-42d6-8959-62c6f8481dbe)


The second model has a balanced accuracy rate of 68.3%. The prediction value for an employee not leaving the company is .78, while the prediction value for an employee leaving the company is .53. The model has gotten better at predicting that an employee will be leaving but has gotten worse at predicting they will stay. This model is better because while there may be a few more falsey predicted employees leaving, more actual employees who are leaving will be identified. I believe a false positive in this siutation is better than a false negative. 

## Neural Network
The first neural network I initiated had 2 hidden layers, the first being the input layer with 26 inputs and nodes, the second having 50 nodes. I fit the model with 150 epochs.
![nn_1](https://github.com/wanderfarther/machine-learning-project/assets/132155105/811715f1-392d-4bfa-8275-a07afcadcdea)
![nn_1_summary](https://github.com/wanderfarther/machine-learning-project/assets/132155105/c70e751c-34d5-447e-9d77-b5ac5cb84420)
![nn_1_evaluation](https://github.com/wanderfarther/machine-learning-project/assets/132155105/660d204a-54ed-4a9b-bbe6-8b4b414358d6)


Around epoch 80, the accuracy and loss started to plateau until epoch 127 where it flutuated between .77 and .78 accuracy. This paired with the final accuracy of .70 and loss of .87 told me this was a bad model.

### Neural Network Optemization
I added an additional layer to the network, but reduced the total number of nodes. The input layer has 26 nodes, the second layer has 30 and the third has 10 nodes. I also reduced the number of epochs from 150 to 50 to avoid overfitting.
![nn_2](https://github.com/wanderfarther/machine-learning-project/assets/132155105/14b63d9b-a772-428b-b4fe-72aa76f18cd8)
![nn_2_summary](https://github.com/wanderfarther/machine-learning-project/assets/132155105/83b00ed6-d6d3-418b-825f-d0c51971c6d4)
![nn_2_evaluation](https://github.com/wanderfarther/machine-learning-project/assets/132155105/837ed3a1-dd8c-409e-b755-7c7cedae394d)

The network's final accuracy is .76 with a loss of .59. This model does not have a particularly good accuracy and a high number of errors. I would choose the second linear regression model over the neural networks.

## Testing Best Model on Department of Labor Dataset
I completed the same data cleaning measures that were done on the Department of Commerce dataset previously in the notebook.
![logistic_regression_labor](https://github.com/wanderfarther/machine-learning-project/assets/132155105/20f28cc7-b919-4f42-b59c-a96eadde2c86)
![logistic_regression_labor_predictions](https://github.com/wanderfarther/machine-learning-project/assets/132155105/769c44e6-038a-4066-8230-ec6057768e28)

The model had an accuracy score of 69%, which is slightly higher than the model testing data. The f-scores are the same for predicting if an employee is not leaving, but is slightly better at a .59 for predicting that an employee is leaving. 

# Summary
The best model to use with this data is the second, optimized logsitic regression model. With more study, a higher accuracy rate may have been achieved. The highest accuracy rate I was able to achieve was 76% but that was in a neural network that had a high possibility of erros. I chose to go with a model that is less accurate, but has better prediction scores (f-scores). The features that affect the predicitions the most are the self-reflective features that inlcude how connected an employee feels to their job, how useful they feel at work, and how much they feel they are allowed to accomplish. Creating an environment that encouirages employees and helps them continue to feel necessary and that their talents are well utlizied is the best way to keep employee retention high. 

