## Rakshitha Mahesh (rxm210063)
## Jeevan desouza (jxd210021)
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression



from google.colab import drive
drive.mount('/content/gdrive')

wine_quality = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

print("Info:-")
wine_quality.info()
print("Describe:-")
wine_quality.describe(include='all')

sns.set(rc={'figure.figsize':(10,8)})
sns.distplot(wine_quality['quality'], bins=30)
plt.show()

#check for null values
wine_quality.isnull().sum()
#check for duplicate values and remving them
print(wine_quality.shape)
wine_quality = wine_quality.drop_duplicates()
wine_quality= wine_quality.dropna()
print(wine_quality.shape)




correlation_matrix = wine_quality.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

# from correlation matrix we can remove x1 transaction date

X = wine_quality[["fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"]]
Y = wine_quality["quality"]

print("------Data Standardization------")
sc = StandardScaler()
df_nor = sc.fit_transform(wine_quality)
df_nor=pd.DataFrame(df_nor)

# Pairwise Correlation for columns
df_nor.corr()
df_nor.columns = wine_quality.columns

print("df_nor->")
print(df_nor)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, 
                                                    test_size = 0.2, random_state = 100)

# Cost Function
def calculate_cost(input_x, output_y, theta):
  num_samples = len(input_x)
  cost = np.sum((input_x.dot(theta) - output_y ) ** 2)/(2 * num_samples)
  return cost

# Gradient Descent Function
def lin_reg_grad_descent(input_x, output_y, theta, alpha, max_iterations):
 cost_history = [0] * max_iterations
 num_samples = len(input_x)
 
 for iteration in range(max_iterations):
    hypothesis = input_x.dot(theta)
    loss = hypothesis - output_y
    gradient = input_x.T.dot(loss) / num_samples
    theta = theta - alpha * gradient
    cost = calculate_cost(input_x, output_y, theta)
    cost_history[iteration] = cost
 
 return theta, cost_history

# Feature Split - Attribute Selection
x=df_nor[['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']]
y=df_nor.iloc[:,11]

# Split Training and Testing Data by percentage:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
alpha = 0.4 #Learning Rate
max_iter = 4000 #Epoch

# Train Data
B = np.zeros(x_train.shape[1])
Bnew, cost_history = lin_reg_grad_descent(x_train, y_train, B, alpha, max_iter)
print(Bnew)
print(cost_history)

# Cost/Loss Plot per iteration
plt.plot(cost_history)
plt.xlabel("Epoch(Number of Iterations)")
plt.ylabel("Cost(Loss)")
plt.show()
print(f'Epoch = {max_iter}')
print(f'Learning Rate(Alpha) = {alpha}')
print(f'Lowest cost = {str(np.min(cost_history))}')
print('-------------------------------------------------------------')
print(f'Cost after {max_iter} iterations = {str(cost_history[-1])}')

# Predict Output for Test Data
y_predicted= np.dot(x_test, Bnew)

# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Training
y_pred_train = np.dot(x_train, Bnew)
print("------Training Performance Evaluation-------")
print("Mean Absolute Error(MAE)-",mean_absolute_error(y_train,y_pred_train))
print("Mean Squared Error(MSE)-",mean_squared_error(y_train,y_pred_train))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_train,y_pred_train)))
print("R2-",r2_score(y_train,y_pred_train))

print("------Testing Performance Evaluation-------")
print("Mean Absolute Error (MAE)-",mean_absolute_error(y_test,y_predicted))
print("Mean Square Error (MSE)-",mean_squared_error(y_test,y_predicted))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_test,y_predicted)))
print("R2-",r2_score(y_test,y_predicted))