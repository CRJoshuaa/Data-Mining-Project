import streamlit as st

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

def get_day_col(df):
    dayCol=pd.DatetimeIndex(df['Date']).day
    return df.assign(Day=dayCol)

def get_month_col(df):
    monthCol=pd.DatetimeIndex(df['Date']).month
    return df.assign(Month=monthCol)

def get_time_of_day(df):
    timeOfDay=pd.cut(pd.DatetimeIndex(df['Time']).hour, bins=[-1,4,11,15,19,23], labels=["Midnight","Morning", "Afternoon", "Evening","Night"]).astype(str)
    return df.assign(Time_Of_Day=timeOfDay)

def get_hour(df):
    hour = pd.DatetimeIndex(df['Time']).hour
    return df.assign(Hour=hour)

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
    shirt_t = df['shirt_type'].apply(lambda x: "{}{}".format('S_', x))
    shirt_c = df['Shirt_Colour'].apply(lambda x: "{}{}".format('S_', x))
    return df.assign(shirt_type=shirt_t,Shirt_Colour=shirt_c)

def mark_pants(df):
    pants_t = df['pants_type'].apply(lambda x: "{}{}".format('P_', x))
    pants_c = df['Pants_Colour'].apply(lambda x: "{}{}".format('P_', x))
    return df.assign(pants_type=pants_t,Pants_Colour=pants_c)

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

cluster_select=['Time_Of_Day','Hour','Race','Gender','Age_Group','Age_Range','Body_Size','With_Kids','Kids_Category','Basket_Size']

classifier1_select=['Time_Of_Day','Race','Gender','Body_Size','With_Kids','Kids_Category','Basket_Size','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Washer_No','Dryer_No']

#############SIDEBAR####################
testSplit=st.sidebar.slider("Test Set Split",0.1,0.9)

#############DATASET PIPING####################
cleaned=(df.pipe(change_to_date)
        .pipe(get_day_col)
        .pipe(get_month_col)
        .pipe(get_hour)
        .pipe(fill_age)
        .pipe(fill_withKids_yes)
        .pipe(fill_withKids_no)
        .pipe(drop_no)
        .pipe(fill_null_val)
        .pipe(mark_washer)
        .pipe(mark_dryer)
        .pipe(mark_shirt)
        .pipe(mark_pants)
        .pipe(get_time_of_day)
        .pipe(bin_age)
    )

st.title('Laundry Service Data Mining')

st.markdown("## Original dataset")
st.write(df)
st.write("Shape of data: ",df.shape)
# st.markdown("### List of Columns")
# st.write(df.columns)

st.markdown("## Processed Dataset")
st.write(cleaned)
st.write("Shape of data: ",cleaned.shape)
# st.markdown("### List of Columns")
# st.write(cleaned.columns)

#############EDA####################
st.markdown("## Exploratory Data Analytics")

###Age of Customers#####
st.markdown("### Analysis of the Ages of Customers")
h = cleaned['Age_Range'].tolist()
h.sort()

hmin=np.min(h)
hmean = np.mean(h)
hmax = np.max(h)
hstd = np.std(h)

pdf = stats.norm.pdf(h, hmean, hstd)

plt.figure(figsize=(15,5))
plt.subplot(121),plt.plot(h, pdf),plt.title('Normal Distribution of the Age Range of Customers')
plt.subplot(122),sns.boxplot(cleaned['Age_Range']),plt.title('Boxplot of the Age Range of Customers'),plt.xlabel('Age of Customers')
st.pyplot()

st.write('Min: ',hmin)
st.write('Mean: ',hmean)
st.write('Max: ',hmax)
st.write('Standard Deviation: ',hstd)

###Race of individuals analysis#####
plt.figure(figsize=(10,5))
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

###Hourly Analysis#####
st.markdown("### Analysis of Time")
hours=pd.to_datetime(df['Time']).dt.hour
hours.hist(bins = 23, range=[0,23], facecolor='green')
plt.title ('Preferable Hour Of Visit')
plt.xlabel('Hour')
plt.ylabel('Number of Customers')
st.pyplot()

#############CLUSTERING####################
st.markdown("## Clustering")
cluster=(cleaned.pipe(select_cluster))

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

plt.plot(range(1, 11), distortions, marker='o',color='orange')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Analysis')
st.pyplot()

k=st.slider("No. of Clusters",2,10)
km=KMeans(n_clusters=k,random_state=1)
km.fit(cluster_dum)
cluster_vis=cluster.copy()
cluster_vis['label']=km.labels_

sns.boxplot(x="label", y="Age_Range", data=cluster_vis)
plt.xlabel('Cluster Labels')
plt.ylabel('Age of Customers')
plt.title('Ages of Clusters')
st.pyplot()

sns.boxplot(x="label", y="Hour", data=cluster_vis)
plt.xlabel('Cluster Labels')
plt.ylabel('Hours visited')
plt.title('Hours Visited by Clusters')
st.pyplot()
#############CLASSIFICATIONS####################
st.markdown("## Classification Models")

###Basket Size#####
st.markdown("### Basket Size Classification")

X=cleaned[['Race','Basket_colour','Pants_Colour','Shirt_Colour','Attire','Washer_No','Time_Of_Day','Dryer_No','Body_Size','Wash_Item']]
X=X.apply(LabelEncoder().fit_transform)

y=cleaned['Basket_Size']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSplit, random_state = 1)

clf = SVC(kernel='linear',gamma='auto')
clf=clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

mcm=multilabel_confusion_matrix(y_test, y_pred,labels=["big", "small", "Unknown"])
st.write("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
st.write("Precision= {:.2f}".format(precision_score(y_test,y_pred, average='micro')))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, average='micro')))

plt.figure(figsize=(15,5))
plt.subplot(131),sns.heatmap(mcm[0], square=True, annot=True,fmt= '.0f'),plt.title('Confusion Matrix for Big')
plt.subplot(132),sns.heatmap(mcm[1], square=True, annot=True,fmt= '.0f'),plt.title('Confusion Matrix for Small')
plt.subplot(133),sns.heatmap(mcm[2], square=True, annot=True,fmt= '.0f'),plt.title('Confusion Matrix for Unknown')
st.pyplot()
# raceChoice=st.selectbox('Race',cleaned['Race'].unique())
# bodySizeChoice=st.selectbox('Body_Size',cleaned['Body_Size'].unique())
# timeOfDayChoice=st.selectbox('Time_Of_Day',cleaned['Time_Of_Day'].unique())
# basketColorChoice=st.selectbox('Basket_colour',cleaned['Basket_colour'].unique())
# pantsColorChoice=st.selectbox('Pants_Colour',cleaned['Pants_Colour'].unique())
# shirtColorChoice=st.selectbox('Shirt_Colour',cleaned['Shirt_Colour'].unique())
# attireChoice=st.selectbox('Attire',cleaned['Attire'].unique())
# washerNoChoice=st.selectbox('Washer_No',cleaned['Washer_No'].unique())
# dryerNoChoice=st.selectbox('Dryer_No',cleaned['Dryer_No'].unique())
# washItemChoice=st.selectbox('Wash_Item',cleaned['Wash_Item'].unique())

###With Kids#####
st.markdown("### With Kids Classification")
X=cleaned[['Race','Basket_colour','Pants_Colour','Shirt_Colour','Attire','Washer_No','Time_Of_Day','Dryer_No','Body_Size','Wash_Item']]
X=X.apply(LabelEncoder().fit_transform)

y=cleaned['With_Kids']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSplit, random_state = 1)

withkidsrf=DecisionTreeClassifier(criterion='entropy',max_depth=10)
withkidsrf=withkidsrf.fit(X, y)
y_pred = withkidsrf.predict(X_test)

st.write("Accuracy on test set: {:.3f}".format(withkidsrf.score(X_test, y_test)))
st.write("Precision= {:.2f}".format(precision_score(y_test,y_pred, average='micro')))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, average='micro')))
mcm=multilabel_confusion_matrix(y_test, y_pred,labels=["yes", "no", "Unknown"])

plt.figure(figsize=(15,5))
plt.subplot(131),sns.heatmap(mcm[0], square=True, annot=True,fmt= '.0f'),plt.title('Confusion Matrix for Yes')
plt.subplot(132),sns.heatmap(mcm[1], square=True, annot=True,fmt= '.0f'),plt.title('Confusion Matrix for No')
plt.subplot(133),sns.heatmap(mcm[2], square=True, annot=True,fmt= '.0f'),plt.title('Confusion Matrix for Unknown')
st.pyplot()
