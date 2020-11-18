#plot_gp(mu_s, cov_s, np.array(X_pred[["on_velocity"]]), X_train=X, Y_train=y)
import os
import pandas as pd
import numpy as np
import json
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model             import LinearRegression
from sklearn.linear_model             import LogisticRegression
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.neighbors                import KNeighborsClassifier
from sklearn.discriminant_analysis    import LinearDiscriminantAnalysis
from sklearn.naive_bayes              import GaussianNB
from sklearn.svm                      import SVC
from sklearn.model_selection          import train_test_split
from sklearn                          import preprocessing
from sklearn                          import metrics
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

# Load data
data_folder = r"C:\Users\User\git\robo_piano_learning\data\piano_learning_experiment_47"
#data_folder = r"C:\Users\chery\Desktop\Workspace\robo_piano_learning\data\piano_learning_experiment_47"
demo_data = r"C:\Users\User\git\robo_piano_learning\data\piano_learning_experiment_50"

input_param_dir = os.path.join(data_folder, r"all_input_parameters.json")
output_param_dir = os.path.join(data_folder, r"all_output_parameters.json")
demo_dir = os.path.join(demo_data, r"demo_parameters.json")

# load parameter data
with open(input_param_dir) as input_param_file:
    input_param = pd.DataFrame(json.load(input_param_file))
with open(output_param_dir) as output_param_file:
    output_param = pd.DataFrame(json.load(output_param_file))
with open(demo_dir) as demo_file:
    demo_param = pd.DataFrame(json.load(demo_file))

# preprocess data
input_param = input_param.drop(columns=['wait_time'])
output_param = output_param.drop(columns=["note"])
input_param = input_param.rename(columns= {"on_velocity": "Q1", #downwards velocity
                                           "off_velocity": "Q2", #upwards velocity
                                           "hold_time": "Q3"}) #wait time between velocities

df = pd.concat([input_param, output_param], axis=1, sort=False)
#print(df)

## remove outliers - might delete
#df = df[df['off_velocity'] > 20] 
#df = df[df['hold_time'] < 5] 

# only select data for 1st note 
#df = df[:3000]

#df = df[df["note"] == df.note.unique()[0]]
##df = df[df["Q1"] == df.Q1.unique()[0]]
#df = df[df["Q2"] == df.Q2.unique()[0]]
#df = df[df["Q3"] == df.Q3.unique()[0]]


print(df)
sns.relplot(x="on_velocity", y="Q1", hue="note", data=df)
#sns.catplot(x="on_velocity", y="Q2", hue="Q1", data=df)
#sns.catplot(x="on_velocity", y="Q3", hue="Q1", data=df)
#sns.catplot(x="off_velocity", y="Q1", hue="Q1", data=df)
#sns.catplot(x="off_velocity", y="Q2", hue="note", data=df)
#sns.catplot(x="off_velocity", y="Q3", hue="Q3", data=df)
#sns.catplot(x="hold_time", y="Q1", hue="Q1", data=df)
#sns.catplot(x="hold_time", y="Q2", hue="Q2", data=df)
#sns.catplot(x="hold_time", y="Q3", hue="note", data=df)

plt.show()

X = df[["on_velocity","off_velocity","hold_time"]]
Q1y = df["Q1"]
Q2y = df["Q2"]
Q3y = df["Q3"]

"""Linear Regression Model"""
#X_train, X_test, y_train, y_test = train_test_split(X, df["Q1"], test_size=0.2, random_state=0)
#
# Encode to distinct classes for non-linear models
# lab_enc = preprocessing.LabelEncoder()
# y = lab_enc.fit_transform(y)
# y = (y*10**5).astype(int)
#



print("regression")
regressor = LinearRegression()
Q1_r = regressor.fit(X,Q1y)
Q1y_pred = Q1_r.predict(demo_param[["on_velocity","off_velocity","hold_time"]])
print(Q1_r.coef_*10**3)

Q2_r = regressor.fit(X,Q2y)
Q2y_pred = Q2_r.predict(demo_param[["on_velocity","off_velocity","hold_time"]])
print(Q2_r.coef_*10**3)

Q3_r = regressor.fit(X,Q3y)
Q3y_pred = Q3_r.predict(demo_param[["on_velocity","off_velocity","hold_time"]])
print(Q3_r.coef_*10**3)

y_pred = pd.DataFrame({"Q1":Q1y_pred,"Q2":Q2y_pred,"Q3":Q3y_pred})

print(demo_param)
print(Q1y_pred)
print(Q2y_pred)
print(Q3y_pred)
print(y_pred)





# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
# print("Coefficients: ", coeff_df)
#
# y_pred = regressor.predict(X_test)
#
# df = pd.DataFrame({'Actual': y_test_on_vel, 'Predicted': y_pred_on_vel})
# df1 = df.head(25)
#
# df1.plot(kind='bar',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""Gaussian Process Model"""
# #rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
# gpr = GaussianProcessRegressor(kernel=kernel)
# gpr.fit(X, y)
# X_pred = demo_param[["on_velocity","off_velocity","hold_time"]]
# y_pred, sigma = gpr.predict(X_pred,return_std=True)
# print(X_pred)
# print(y_pred)
# print(sigma)

##gpr = GaussianProcessRegressor(kernel=rbf)
#gpr.fit(X, y)
#X_pred = demo_param[["on_velocity","off_velocity","hold_time"]]
## Compute posterior predictive mean and covariance
#mu_s, cov_s = gpr.predict(X_pred, return_cov=True)
#
## Obtain optimized kernel parameters
#l = gpr.kernel_.k2.get_params()['length_scale']
#sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])
#
## Compare with previous results
##assert(np.isclose(l_opt, l))
##assert(np.isclose(sigma_f_opt, sigma_f))
#
## Plot the results
