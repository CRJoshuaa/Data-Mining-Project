import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

df=pd.read_csv('Laundry_Data.csv')
############ClEANING#########################
def fill_null_val(df):
    return df.fillna('Unknown')

def change_to_date(df):
    date=pd.to_datetime(df['Date'], infer_datetime_format=True)
    return df.assign(Date=date)

def change_to_time(df):
    time=pd.to_datetime(df['Time'], infer_datetime_format=True)
    time=pd.DatetimeIndex(time).time
    return df.assign(Time=time)

def get_day_col(df):
    dayCol=pd.DatetimeIndex(df['Date']).day
    return df.assign(Day=dayCol)

def get_month_col(df):
    monthCol=pd.DatetimeIndex(df['Date']).month
    return df.assign(Month=monthCol)

def get_month_col(df):
    monthCol=pd.DatetimeIndex(df['Date']).month
    return df.assign(Month=monthCol)

def get_time_of_day(df):
    timeOfDay=pd.cut(pd.DatetimeIndex(df['Time']).hour, bins=[-1,4,11,15,19,23], labels=["Midnight","Morning", "Afternoon", "Evening","Night"])
    return df.assign(Time_Of_Day=timeOfDay)

def fill_age(df):
    age=df['Age_Range'].fillna(round(df['Age_Range'].mean()))
    age=age.astype('int64')
    return df.assign(Age_Range=age)

def fill_withKids_yes(df):
    with_kids = np.where(df["Kids_Category"]=='no_kids',df['With_Kids'],'yes')
    return df.assign(With_Kids=with_kids)

def drop_no(df):
    return df.drop('No',axis=1)

def fill_withKids_no(df):
    with_kids = np.where(df["Kids_Category"]!='no_kids',df['With_Kids'],'no')
    return df.assign(With_Kids=with_kids)

def W6_big(df):
    washer = np.where(df["Washer_No"]!=6,df['Basket_Size'],'big')
    return df.assign(Basket_Size=washer)

def bin_age(df):
    age_group= pd.cut(x=df['Age_Range'], bins=[20, 30, 40, 50,60])
    return df.assign(Age_Group=age_group)




def mark_washer(df):
    washer = df['Washer_No'].apply(lambda x: "{}{}".format('W_', x))
    return df.assign(Washer_No=washer)

def mark_dryer(df):
    dryer = df['Dryer_No'].apply(lambda x: "{}{}".format('D_', x))
    return df.assign(Dryer_No=dryer)

def mark_shirt(df):
    shirt = df['shirt_type'].apply(lambda x: "{}{}".format('S_', x))
    return df.assign(shirt_type=shirt)

def mark_pants(df):
    pants = df['pants_type'].apply(lambda x: "{}{}".format('P_', x))
    return df.assign(pants_type=pants)

arm_drop=['Date','Time','Age_Range','Race','Gender','Body_Size','With_Kids','Kids_Category','Spectacles']

arm_select1=['Time_Of_Day','Basket_Size','Basket_colour','Washer_No','Dryer_No','Wash_Item']
arm_select2=['Time_Of_Day','Gender','Body_Size','Age_Group','Attire','Kids_Category','Spectacles']


cluster_select=['Time_Of_Day','Race','Gender','Age_Group','Age_Range','Body_Size','With_Kids','Kids_Category','Basket_Size']

classifier1_select=['Time_Of_Day','Race','Gender','Body_Size','With_Kids','Kids_Category','Basket_Size','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Washer_No','Dryer_No']
#####################################################################################################################################

def drop_arm(df):
    return df.drop(columns=arm_drop)

def select_arm1(df):
    return df[arm_select1]

def select_arm2(df):
    return df[arm_select2]


def select_cluster(df):
    return df[cluster_select]

def select_classifier1(df):
    return df[classifier1_select]

def get_dataset(dataset_name):
    if dataset_name=='Iris':
        df=datasets.load_iris()
        #df=pd.read_csv('Laundry_Data.csv')
    elif dataset_name=='Breast Cancer':
        df=datasets.load_breast_cancer()
    else:
        df=datasets.load_wine()

    X=df.data
    y=df.target

    return X,y


def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=='KNN':
        K=st.sidebar.slider("K",1,15)
        params['K']=K
    elif clf_name=='SVM':
        C=st.sidebar.slider("C",0.01,10.0)
        params['C']=C
    else:
        max_depth=st.sidebar.slider("Max_depth",2,15)
        n_estimators=st.sidebar.slider("N_estimators",1,100)
        params['max_depth']=max_depth
        params['n_estimators']=n_estimators

    return params

def get_classifier(clf_name,params):
    if clf_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name=='SVM':
        clf=SVC(C=params['C'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],
                                    max_depth=params['max_depth'],
                                    random_state=1)

    return clf



#############DATASET PIPING####################
cleaned=(df.pipe(change_to_date)
        .pipe(get_day_col)
        .pipe(get_month_col)
        .pipe(fill_age)
        .pipe(fill_withKids_yes)
        .pipe(fill_withKids_no)
        .pipe(drop_no)
        .pipe(fill_null_val)
    )
#side bar
dataset_name=st.sidebar.selectbox("Select Dataset",('Iris','Breast Cancer','Wine'))
classifier_name=st.sidebar.selectbox("Select Model",('KNN','SVM','Random Forest','Gaussian'))


st.title('Laundry Service Data Mining')

st.markdown("## The dataset")
st.write(cleaned)
X,y=get_dataset(dataset_name)
st.write("Shape of data",X.shape)
st.write('Number of classes',len(np.unique(y)))



st.markdown("## Exploratory Data Analytics")


###Race of individuals analysis#####
st.markdown("### Analysis of the Races")
ax=sns.countplot(data = cleaned, x = 'Race')
st.pyplot()

###With Kids Analysis#####
st.markdown("### Analysis of Kids")
ax=sns.countplot(data = cleaned, y = 'With_Kids')
total = len(df['With_Kids'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
st.pyplot()

st.markdown("## Classification Models")

# params=add_parameter_ui(classifier_name)
#
# clf=get_classifier(classifier_name,params)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)
#
# acc=accuracy_score(y_test,y_pred)
# st.write(f'classifier={classifier_name}')
# st.write(f'accuracy={acc}')
#
# #PLOT
# pca=PCA(2)
# X_projected=pca.fit_transform(X)
#
# x1=X_projected[:,0]
# x2=X_projected[:,1]
#
# fig=plt.figure()
# plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
# plt.xlabel('Principle Component 1')
# plt.ylabel('Principle Component 2')
# plt.colorbar()
#
# st.pyplot(plt)










# st.success("Successful")
# st.info("Information")
# st.warning("Warning")
# st.error('ERROR')
# st.exception("TypeERROR('Name not found')")
# st.write(1+2)
