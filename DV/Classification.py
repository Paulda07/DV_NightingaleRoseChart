import pandas as pd
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime

def classification():
    df = pd.read_csv(r'C:\Users\pauls\Documents\Part1_Crime_data.csv')

    df["Inside_Outside"].replace({"Inside": "I", "Outside": "O"}, inplace=True)
    print(df.shape)
    print(df['Inside_Outside'].unique())



    df = df[['Description', 'Weapon', 'District', 'Premise', 'Inside_Outside']]
    # print(data.shape)

    #Getting data for classification, we will use NAN data later
    data = df.dropna(subset=['Inside_Outside'])
    plotting(data)
    initial_df = data

    data = data.values
    X = data[:,:-1].astype(str)
    y = data[:, -1].astype(str)
    # print (X)
    # print(y)
    # print('Input', X.shape)
    # print('Output', y.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # ordinal encode input variables
    ordinal_encoder = OneHotEncoder(handle_unknown='ignore')
    # ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value')
    ordinal_encoder.fit(X_train)
    X_train = ordinal_encoder.transform(X_train)
    X_test = ordinal_encoder.transform(X_test)
    # ordinal encode target variable
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    # define the model
    # model = LogisticRegression()#- 62.32%, 97.92
    model = DecisionTreeClassifier()#75.38 98.12, 98.11
    # model = KNeighborsClassifier()#71.81
    # model = LinearDiscriminantAnalysis() #62.48
    # model = GaussianNB() 54.56
    # model = SVC()
    # fit on the training set
    model.fit(X_train, y_train)
    # predict on test set
    yhat = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.2f' % (accuracy*100))

    # print(np.unique(yhat))

    df = df[df['Inside_Outside'].isna()]

    df = df.values
    X_df = df[:,:-1].astype(str)
    df_new = ordinal_encoder.transform(X_df)
    pred = model.predict(df_new)
    print(pred)

    final_df = pd.DataFrame(X_df, columns = ['Description', 'Weapon', 'District', 'Premise'])
    final_df_label = pd.DataFrame (pred, columns = ['Inside_Outside'])
    final_df_label = pd.concat([final_df,final_df_label], axis = 1)
    final_df_label["Inside_Outside"].replace({1: "I", 0: "O"}, inplace=True)
    print(final_df_label['Inside_Outside'].unique())
    final_df = pd.concat([final_df_label, initial_df])

    # print(final_df.shape)
    print(final_df['Inside_Outside'].unique())
    return final_df


def plotting(data):
    df_grouped=data.groupby(['Description', 'Inside_Outside']).size().reset_index(name='Counts')
    df_inside = df_grouped[df_grouped["Inside_Outside"] == 'I']
    df_outside = df_grouped[df_grouped["Inside_Outside"] == 'O']

    x = df_inside['Description'].values

    x_len = np.arange(len(x))
    y = df_inside['Counts'].values

    z = df_outside['Counts'].values

    ax = plt.subplot(111)
    ax.bar(x_len-0.1, y, width=0.2, color='b', align='center')
    ax.bar(x_len+0.1, z, width=0.2, color='g', align='center')
    # ax.bar(x_len+0.2, k, width=0.2, color='r', align='center')
    # ax.xticks(x_len, x)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=2)
    ax.set_xticks(x_len)
    ax.set_xticklabels(x)
    ax.autoscale(tight=True)
    ax.legend(['Inside','Outside'])
    plt.show()
    # print (df_inside, df_outside)

        



data = classification()
plotting(data)