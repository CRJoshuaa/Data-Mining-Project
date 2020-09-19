import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    timeOfDay=pd.cut(pd.DatetimeIndex(df['Time']).hour, bins=[-1,4,11,15,19,23], labels=["Midnight","Morning", "Afternoon", "Evening","Night"]).astype(str)
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
    age_group= pd.cut(x=df['Age_Range'], bins=[20, 30, 40, 50,60]).astype(str)
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


arm_drop=['Date','Time','Age_Range','Race','Gender','Body_Size','With_Kids','Kids_Category','Spectacles']

arm_select1=['Time_Of_Day','Basket_Size','Basket_colour','Washer_No','Dryer_No','Wash_Item']
arm_select2=['Time_Of_Day','Gender','Body_Size','Age_Group','Attire','Kids_Category','Spectacles']


cluster_select=['Time_Of_Day','Race','Gender','Age_Group','Age_Range','Body_Size','With_Kids','Kids_Category','Basket_Size']

classifier1_select=['Time_Of_Day','Race','Gender','Body_Size','With_Kids','Kids_Category','Basket_Size','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Washer_No','Dryer_No']
#####################################################################################################################################

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
        .pipe(mark_washer)
        .pipe(mark_dryer)
        .pipe(drop_no)
        .pipe(fill_null_val)
        .pipe(get_time_of_day)
        .pipe(bin_age)
    )


#side bar
testSplit=st.sidebar.slider("Test Set Split",0.1,0.9)
dataset_name=st.sidebar.selectbox("Select Dataset",('Iris','Breast Cancer','Wine'))
classifier_name=st.sidebar.selectbox("Select Model",('KNN','SVM','Random Forest','Gaussian'))


st.title('Laundry Service Data Mining')

st.markdown("## Original dataset")
st.write(df)
st.write("Shape of data: ",df.shape)
st.markdown("### List of Columns")
st.write(df.columns)

st.markdown("## Processed Dataset")
st.write(cleaned)
st.write("Shape of data: ",cleaned.shape)
st.markdown("### List of Columns")
st.write(cleaned.columns)

st.markdown("## Exploratory Data Analytics")

###Race of individuals analysis#####
st.markdown("### Analysis of the Races")
ax=sns.countplot(data = cleaned, x = 'Race')
plt.title ('Customer Visits by Race')
plt.xlabel('Races')
plt.ylabel('Number of Customers')

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

plt.title ('Presence of Kids With Customers')
plt.xlabel('Number of Customers')
plt.ylabel('Presence of Kids')
st.pyplot()

st.markdown("### Analysis of Time")
hours=pd.to_datetime(df['Time']).dt.hour
hours.hist(bins = 23, range=[0,23], facecolor='green')
plt.title ('Preferable Hour Of Visit')
plt.xlabel('Hour')
plt.ylabel('Number of Customers')
st.pyplot()

###############################################################################
st.markdown("## Clustering")
cluster=(df.pipe(change_to_date)
        .pipe(fill_age)
        .pipe(fill_withKids_yes)
        .pipe(fill_withKids_no)
        .pipe(fill_null_val)
        .pipe(get_time_of_day)
        .pipe(bin_age)
        .pipe(select_cluster)
    )

cluster_dum=pd.get_dummies(cluster,drop_first=True)

distortions = []
for i in range(1,11):
    km=KMeans(
        n_clusters=i,
        init='random',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=0
    )
    km.fit(cluster_dum)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Analysis')
st.pyplot()


km=KMeans(n_clusters=3,random_state=1)
km.fit(cluster_dum)
cluster_vis=cluster.copy()
cluster_vis['label']=km.labels_

#ax = sns.relplot(x="Age_Range", y="Time_Of_Day", hue=cluster_vis.label.tolist(), data=cluster_vis)
sns.boxplot(x="label", y="Age_Range", data=cluster_vis)
plt.xlabel('Cluster Labels')
plt.ylabel('Age of Customers')
plt.title('Ages of Clusters')
st.pyplot()

##################################################################################
st.markdown("## Classification Models")
st.markdown("### Basket Size Classification")

X=cleaned[['Race','Basket_colour','Pants_Colour','Shirt_Colour','Attire','Washer_No','Time_Of_Day','Dryer_No','Body_Size','Wash_Item']]
X=X.apply(LabelEncoder().fit_transform)

y=cleaned['Basket_Size']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSplit, random_state = 1)

clf = SVC(kernel='linear',gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.write("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

raceChoice=st.selectbox('Race',cleaned['Race'].unique())
bodySizeChoice=st.selectbox('Body_Size',cleaned['Body_Size'].unique())
timeOfDayChoice=st.selectbox('Time_Of_Day',cleaned['Time_Of_Day'].unique())
basketColorChoice=st.selectbox('Basket_colour',cleaned['Basket_colour'].unique())
pantsColorChoice=st.selectbox('Pants_Colour',cleaned['Pants_Colour'].unique())
shirtColorChoice=st.selectbox('Shirt_Colour',cleaned['Shirt_Colour'].unique())
attireChoice=st.selectbox('Attire',cleaned['Attire'].unique())
washerNoChoice=st.selectbox('Washer_No',cleaned['Washer_No'].unique())
dryerNoChoice=st.selectbox('Dryer_No',cleaned['Dryer_No'].unique())
washItemChoice=st.selectbox('Wash_Item',cleaned['Wash_Item'].unique())

st.markdown("### With Kids Classification")
X=cleaned[['Race','Basket_colour','Pants_Colour','Shirt_Colour','Attire','Washer_No','Time_Of_Day','Dryer_No','Body_Size','Wash_Item']]
X=X.apply(LabelEncoder().fit_transform)

y=cleaned['With_Kids']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSplit, random_state = 1)




# mcm=multilabel_confusion_matrix(y_test, y_pred,labels=["big", "small", "Unknown"])
# st.write(mcm)
# st.write(classification_report(y_test,y_pred))
# params=add_parameter_ui(classifier_name)
#
# #PLOT
# st.pyplot(plt)










# st.success("Successful")
# st.info("Information")
# st.warning("Warning")
# st.error('ERROR')
# st.exception("TypeERROR('Name not found')")
# st.write(1+2)
