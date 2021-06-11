import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

st.title('Churn Classifier prediction')

model_name = st.sidebar.selectbox('Select Your Model',('LR','RFC','DTC','KNN','SVM'))

st.write(f"## {model_name} Model")

data_train = pd.read_csv("churn-bigml-Train.csv")
data_test = pd.read_csv("churn-bigml-Test.csv")

label_encoder = preprocessing.LabelEncoder() 
data_train['Internationalplan']= label_encoder.fit_transform(data_train['Internationalplan'])
data_train['Voicemailplan']= label_encoder.fit_transform(data_train['Voicemailplan'])
data_train['Churn']= label_encoder.fit_transform(data_train['Churn'])

label_encoder1 = preprocessing.LabelEncoder() 
data_test['Internationalplan']= label_encoder1.fit_transform(data_test['Internationalplan'])
data_test['Voicemailplan']= label_encoder1.fit_transform(data_test['Voicemailplan'])
data_test['Churn']= label_encoder1.fit_transform(data_test['Churn'])

x_train = data_train.drop(['Churn','State'],axis=1)
y_train = data_train.Churn

x_test = data_test.drop(['Churn','State'],axis=1)
y_test = data_test.Churn

st.sidebar.write(f"Shape of Train Data",x_train.shape)
st.sidebar.write(f"Shape of Test Data",x_test.shape)

def add_parameter_ui(m_name):
    params = dict()

    if m_name == 'LR':
        random_state = st.sidebar.slider('random_state', 20, 80)
        params['random_state'] = random_state

    elif m_name == 'RFC':
        n_estimators = st.sidebar.slider('n_estimators', 50, 150)
        params['n_estimators'] = n_estimators
    
    elif m_name == 'DTC':
        max_depth = st.sidebar.slider('max_depth', 20, 80)
        params['max_depth'] = max_depth
    
    elif m_name == 'KNN':
        n_neighbors = st.sidebar.slider('n_neighbors', 2, 15)
        params['n_neighbors'] = n_neighbors

    elif m_name == 'SVM':
        random_state = st.sidebar.slider('random_state', 20, 80)
        params['random_state'] = random_state

    return params

params = add_parameter_ui(model_name)

def get_model_name(m_name, params):
    model = None

    if m_name == 'LR':
        model = LogisticRegression(random_state=params['random_state'])

    elif m_name == 'RFC':
        model = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=42)

    elif m_name == 'DTC':
        model = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=42)

    elif m_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])

    elif m_name == 'SVM':
        model = SVC(probability=True,random_state=params['random_state'])

    return model

model = get_model_name(model_name, params)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)*100
msr = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f'Model Name = ',model.__class__.__name__)
st.write(f'Accuracy =', acc)
st.write(f'Mean Square Root =', msr)


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(f"Confusion Matrix")

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
st.pyplot()
