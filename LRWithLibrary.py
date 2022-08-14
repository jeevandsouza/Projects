
## Rakshitha Mahesh (rxm210063)
## Jeevan desouza (jxd210021)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


## Rakshitha Mahesh (rxm210063)
## Jeevan desouza (jxd210021)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# From the correlation heatmap, we can remove residual sugar, free sulphur dioxide, and pH.
X = wine_quality[["fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"]]
Y = wine_quality["quality"]

# Scaling the dataset to fit the model
X = StandardScaler().fit_transform(X)


# Dividing the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

# fitting model
part2 = LinearRegression()
part2.fit(x_train,y_train)



y_train_predict = part2.predict(x_train)
rmse_part2 =(np.sqrt(mean_squared_error(y_train,y_train_predict)))
print("Train RMSE: ", rmse_part2)



y_test_predict = part2.predict(x_test)
rmse_part2 =(np.sqrt(mean_squared_error(y_test,y_test_predict)))
print("Test RMSE: ", rmse_part2)