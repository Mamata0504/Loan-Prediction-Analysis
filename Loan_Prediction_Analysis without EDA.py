#Loan Prediction Analysis

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,RobustScaler,PowerTransformer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

#1.Reading of Data
def load_data():
	location=r'D:\Velocity Corporate Training\Python\Databases\Finance Project\loan.csv'
	df=pd.read_csv(location)
	print(df.head())
	print(df.info())
	print(df.describe())

	#Remove 'month' from term
	df['term']=df['term'].replace({'36 months':36,'60 months':60})

	# Finding Categorical Columns
	categorical_feature_mask = df.dtypes==object

	# filter categorical columns using mask and turn it into a list
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	print(categorical_cols)

	# Finding Numerical Columns
	numerical_feature_mask = df.dtypes!=object

	# filter categorical columns using mask and turn it into a list
	numerical_cols = df.columns[numerical_feature_mask].tolist()
	print(numerical_cols)
	return df


#2. Data Preprocessing and Feature Engineering
def data_prec(df):
	#Finding Missing Values
	df.isnull().sum()

	#Handling Missing Values
	df['emp_length']=df['emp_length'].fillna(df['emp_length'].mean())
	df['annual_inc']=df['annual_inc'].fillna(df['annual_inc'].mean())
	df['delinq_2yrs']=df['delinq_2yrs'].fillna(df['delinq_2yrs'].mode()[0])
	df['longest_credit_length']=df['longest_credit_length'].fillna(df['longest_credit_length'].mode()[0])
	df['revol_util']=df['revol_util'].fillna(df['revol_util'].median())
	df['total_acc']=df['total_acc'].fillna(df['total_acc'].median())
	
	# Finding Categorical Columns
	categorical_feature_mask = df.dtypes==object
	# filter categorical columns using mask and turn it into a list
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	print(categorical_cols)

	#Label Encoding to convert Categorical Features into Numerical Features
	for c in categorical_cols:
		lbl = LabelEncoder() 
		lbl.fit(list(df[c].values)) 
		df[c] = lbl.transform(list(df[c].values))
		
	#Apply log transformation to deal with skewness of annual income
	df['annual_inc']=np.log1p(df['annual_inc'])
	return df

#Classification Model Building

def class_model(df):
	#Data splitting train and test Data
	x = df.drop('bad_loan', axis = 1)
	y = df.bad_loan
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)

	#Feature Scaling
	PT=PowerTransformer()
	x_train=PT.fit_transform(x_train)
	x_test=PT.fit_transform(x_test)

	#LogisticRegression classification Model without cross Validation
	log = LogisticRegression()
	log.fit(x_train, y_train)
	log_pred = log.predict(x_test)
	
	log_accuracy = metrics.accuracy_score(y_test, log_pred)
	print("Accuracy: ",log_accuracy)

	log_precision=metrics.precision_score(y_test, log_pred,pos_label=0)
	print("Precision: ",log_precision)

	log_recall=metrics.recall_score(y_test, log_pred,pos_label=0)
	print("Recall: ",log_recall)

	log_f1_score= metrics.f1_score(y_test, log_pred,pos_label=0)
	print("F1 Score: ",log_f1_score)

	print("Confusion Matrix:\n",confusion_matrix(y_test,log_pred))
	print("Classification Report:\n",classification_report(y_test,log_pred))

	#LogisticRegression classification Model with cross Validation
	PT1=PowerTransformer()
	x=PT1.fit_transform(x)

	log_cross_val = cross_val_score(log, x, y, cv=10, scoring='accuracy')
	print('Classification Results with cross validation::')
	log_cv_accuracy = log_cross_val.mean()
	print("Accuracy: ",log_cv_accuracy)

	log_cross_val_pre = cross_val_score(log, x, y, cv=10, scoring='precision_macro')
	log_cv_precision = log_cross_val_pre.mean()
	print("Precision: ",log_cv_precision)

	log_cross_val_re = cross_val_score(log, x, y, cv=10, scoring='recall_macro')
	log_cv_recall = log_cross_val_re.mean()
	print("Recall: ",log_cv_recall)

	log_cross_val_f1 = cross_val_score(log, x, y, cv=10, scoring='f1_macro')
	log_cv_f1_score = log_cross_val_f1.mean()
	print("F1 Score: ",log_cv_f1_score)

#Driver Functions
#Calling of Data and EDA functions
df=load_data()

#Calling of Data Preprocessing, Feature Engineering and Model building
df=data_prec(df)
print(df.head(5))
class_model(df)
