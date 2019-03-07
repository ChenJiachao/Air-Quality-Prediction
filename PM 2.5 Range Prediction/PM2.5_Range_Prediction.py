#In this final project, I choose the data from  UCL Machine Learning Repository, named PRSA_data_2010.1.1-2014.12.31
#I first Load the CSV file in to the python, after I roughly look through this CSV file, I found there are many unuseful
#features that I need to delete, and also for the target pm 2.5, there are a lot NA value row that I need to delete
#I clean the data first, and then, I load the data with different range of pm 2.5 values. In each group, I divde them
#in to two targets, and then I used Logistic Regression machine learning method to train the data set. I used the
#same method as we talked in the lecture to train the data set.
#And then, I used the ROC method to find the accuracy of Logistic Regression by find the area
#under the ROC curve, and I draw the ROC curve, and then repeat the steps four times.



import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

with open('PRSA_data_2010.1.1-2014.12.31.csv','r') as df:       # Load the CSV file
    datasheet=csv.reader(df,delimiter=',')
    data_list=list(datasheet)

data_list=data_list[1:len(data_list)]

data=[]

for i in range (0,len(data_list)):
    a=  data_list[i]
    if a[5] != "NA" :                                   # delete the data row which doesnt contain PM 2.5 value
        a=  [x.replace('NW','0') for x in a ]           # Replace the letter to numbers for easier classification
        a = [x.replace('NE', '1') for x in a]
        a = [x.replace('SE', '2') for x in a]
        a = [x.replace('cv', '3') for x in a]
        data.append(a[5:11])

########################################## The first group for good and moderate pm2.5 concentration###########################################
# The second, thrid and fourth will repeat the same steps of the first group
print('1. The case for PM2.5 Concetration between 0-12 (good)  and 12-35 (moderate) : ')
print('\n')
data1=[]

for i in range (0,len(data_list)):
    a=  data_list[i]
    if a[5] != "NA" and float(a[5])<=35:                # delete the data row which doesnt contain PM 2.5 value
        a=  [x.replace('NW','0') for x in a ]           # Replace the letter to numbers for easier classification
        a = [x.replace('NE', '1') for x in a]
        a = [x.replace('SE', '2') for x in a]
        a = [x.replace('cv', '3') for x in a]
        data1.append(a[5:11])

sheet_value1=np.asarray(data1)
sheet_value1 = sheet_value1.astype(np.float)


for i in range(0,len(sheet_value1[:,0])):                # devide the data value into different groups for predication
    if int(sheet_value1[i, 0]) <= 12:
        sheet_value1[i, 0] = int (0)
    if  int(sheet_value1[i,0]) >12:
        sheet_value1[i,0] = int (1)
y_LR1=sheet_value1[:,0]
X_LR1=sheet_value1[:,[1,2,3,4,5]]


X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(X_LR1,y_LR1,test_size=0.2,random_state=0)

sc = StandardScaler()
sc.fit(X_train_LR)

X_train_std_LR = sc.transform(X_train_LR)
X_test_std_LR = sc.transform(X_test_LR)

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std_LR, y_train_LR)

y_pred_LR = lr.predict(X_test_std_LR)                           # Test Score for predication
print('Misclassified samples for Logistic Regression in the Group 1: (good and moderate) : %d' % (y_test_LR != y_pred_LR).sum())
print('Accuracy for Logistic Regression: %.2f' % accuracy_score(y_test_LR, y_pred_LR))
ac1=accuracy_score(y_test_LR, y_pred_LR)

classification_report = metrics.classification_report(y_test_LR, y_pred_LR)  # Showing the classfication report
print('Classification report: ')
print(classification_report)

y_prob_LR = lr.predict_proba(X_test_LR)[:, 1]
y_pred_LR = np.where(y_prob_LR > 0.5, 1, 0)

auc_roc = metrics.classification_report(y_test_LR, y_pred_LR)
false_rate, true_rate, thresholds = roc_curve(y_test_LR, y_prob_LR)
roc_auc = auc(false_rate, true_rate)
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(false_rate, true_rate, color='orange', label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='-')
plt.axis('tight')
plt.ylabel('Truth Rate')
plt.xlabel('False  Rate')
plt.title('Good and Moderate PM2.5')
plt.legend()


print('\n')



########################################## The second group for unhealthy for sensitive group and unhealthy pm2.5 concentration###########################
print('2. The case for PM2.5 Concetration between 35- 56 (unhealh for senstive group )  and 56-150 (unhealth ) : ')
print('\n')
data2=[]
for i in range (0,len(data_list)):
    a2=  data_list[i]
    if a2[5] != "NA" and float(a2[5])>35 and float(a2[5])<=150:
        a2 =  [x.replace('NW','0') for x in a2 ]
        a2 = [x.replace('NE', '1') for x in a2]
        a2 = [x.replace('SE', '2') for x in a2]
        a2 = [x.replace('cv', '3') for x in a2]
        data2.append(a2[5:11])

sheet_value2=np.asarray(data2)
sheet_value2 = sheet_value2.astype(np.float)

for i in range(0,len(sheet_value2[:,0])):
    if int(sheet_value2[i, 0]) <=56 :
        sheet_value2[i, 0] = int(0)
    if  int(sheet_value2[i,0]) >56:
        sheet_value2[i,0] = int(1)
y_LR2=sheet_value2[:,0]
X_LR2=sheet_value2[:,[1,2,3,4,5]]

X_train_LR2, X_test_LR2, y_train_LR2, y_test_LR2 = train_test_split(X_LR2,y_LR2,test_size=0.2,random_state=0)

sc.fit(X_train_LR2)
X_train_std_LR2 = sc.transform(X_train_LR2)
X_test_std_LR2 = sc.transform(X_test_LR2)

lr.fit(X_train_std_LR2, y_train_LR2)
y_pred_LR2 = lr.predict(X_test_std_LR2)

print('Misclassified samples for Logistic Regression in the Group 2: (unhealth for sensitive group and unhealth): %d' % (y_test_LR2 != y_pred_LR2).sum())

print('Accuracy for Logistic Regression: %.2f' % accuracy_score(y_test_LR2, y_pred_LR2))
Class_Report = metrics.classification_report(y_test_LR2, y_pred_LR2)
print('Classification report: ')
print(Class_Report)


lr.fit(X_train_LR2, y_train_LR2)

y_prob_LR2 = lr.predict_proba(X_test_LR2)[:, 1]
y_pred_LR2 = np.where(y_prob_LR2 > 0.5, 1, 0)

auc_roc2 = metrics.classification_report(y_test_LR2, y_pred_LR2)
false_rate2, true_rate2, thresholds2 = roc_curve(y_test_LR2, y_prob_LR2)
roc_auc2 = auc(false_rate2, true_rate2)


plt.subplot(1,2,2)
plt.plot(false_rate, true_rate, color='orange', label='AUC = %0.2f' % roc_auc2)
plt.plot([0, 1], [0, 1], linestyle='-')
plt.axis('tight')
plt.ylabel('Truth Rate')
plt.xlabel('False  Rate')
plt.title('Unheath PM2.5')
plt.legend()

print('\n')







########################################## The third group for unhealthy for very unhealth group and harzard pm2.5 concentration#################
print('3. The case for PM2.5 Concetration between 150- 250 (very unhealth )  and 250-350 ( hazard) : ')
print('\n')
data3=[]
for i in range (0,len(data_list)):
    a3=  data_list[i]
    if a3[5] != "NA" and float(a3[5])>150 and float(a3 [5])<=350:
        a3 =  [x.replace('NW','0') for x in a3 ]
        a3 = [x.replace('NE', '1') for x in a3]
        a3 = [x.replace('SE', '2') for x in a3 ]
        a3 = [x.replace('cv', '3') for x in a3 ]
        data3 .append(a3 [5:11])

sheet_value3=np.asarray(data3)
sheet_value3 = sheet_value3.astype(np.float)

for i in range(0,len(sheet_value3[:,0])):
    if int(sheet_value3[i, 0]) <=250 :
        sheet_value3[i, 0] = int(0)
    if  int(sheet_value3[i,0]) >250:
        sheet_value3[i,0] = int(1)
y_LR3=sheet_value3[:,0]
X_LR3=sheet_value3[:,[1,2,3,4,5]]

X_train_LR3, X_test_LR3, y_train_LR3, y_test_LR3 = train_test_split(X_LR3,y_LR3,test_size=0.2,random_state=0)

sc.fit(X_train_LR3)
X_train_std_LR3 = sc.transform(X_train_LR3)
X_test_std_LR3 = sc.transform(X_test_LR3)


lr.fit(X_train_std_LR3, y_train_LR3)


y_pred_LR3 = lr.predict(X_test_std_LR3)
print('Misclassified samples for Logistic Regression in the Group 3: (very unhealth and hazard): %d' % (y_test_LR3 != y_pred_LR3).sum())
print('Accuracy for Logistic Regression: %.2f' % accuracy_score(y_test_LR3, y_pred_LR3))

Class_Report = metrics.classification_report(y_test_LR3, y_pred_LR3)
print('Classification report: ')
print(Class_Report)

lr.fit(X_train_LR3, y_train_LR3)

y_prob_LR3 = lr.predict_proba(X_test_LR3)[:, 1]
y_pred_LR3 = np.where(y_prob_LR3 > 0.5, 1, 0)

auc_roc3 = metrics.classification_report(y_test_LR3, y_pred_LR3)
false_rate3, true_rate3, thresholds3 = roc_curve(y_test_LR3, y_prob_LR3)
roc_auc3 = auc(false_rate3, true_rate3)

plt.figure(2)
plt.subplot(1,2,1)
plt.title('ROC - GNB')
plt.plot(false_rate3, true_rate3, color='orange', label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.axis('tight')
plt.ylabel('Truth Rate')
plt.xlabel('False  Rate')
plt.title('Uery unheath and Hazard PM2.5')
plt.legend()


print('\n')









########################################## The fourth group for very hazard  pm2.5 concentration#####################################
print('4. The case for PM2.5 Concetration between 350- 500 (very harzard )  and >500 ( above ) : ')
print('\n')
data4=[]
for i in range (0,len(data_list)):
    a4 =  data_list[i]
    if a4 [5] != "NA" and float(a4 [5])>350 :
        a4  =  [x.replace('NW','0') for x in a4  ]
        a4  = [x.replace('NE', '1') for x in a4 ]
        a4  = [x.replace('SE', '2') for x in a4 ]
        a4  = [x.replace('cv', '3') for x in a4  ]
        data4  .append(a4  [5:11])

sheet_value4=np.asarray(data4)
sheet_value4 = sheet_value4.astype(np.float)

for i in range(0,len(sheet_value4[:,0])):
    if int(sheet_value4[i, 0]) <=500 :
        sheet_value4[i, 0] = int(0)
    if  int(sheet_value4[i,0]) >500:
        sheet_value4[i,0] = int(1)

y_LR4=sheet_value4[:,0]
X_LR4=sheet_value4[:,[1,2,3,4,5]]

X_train_LR4, X_test_LR4, y_train_LR4, y_test_LR4 = train_test_split(X_LR4,y_LR4,test_size=0.2,random_state=0)

sc.fit(X_train_LR4)

X_train_std_LR4 = sc.transform(X_train_LR4)
X_test_std_LR4 = sc.transform(X_test_LR4)

lr.fit(X_train_std_LR4, y_train_LR4)

y_pred_LR4 = lr.predict(X_test_std_LR4)
print('Misclassified samples for Logistic Regression in the group 4 (very unhealth): %d' % (y_test_LR4 != y_pred_LR4).sum())
print('Accuracy for Logistic Regression: %.2f' % accuracy_score(y_test_LR4, y_pred_LR4))
Class_Report = metrics.classification_report(y_test_LR4, y_pred_LR4)
print('Classification report: ')
print(Class_Report)
print('\n')

lr.fit(X_train_LR4, y_train_LR4)

y_prob_LR4 = lr.predict_proba(X_test_LR4)[:, 1]
y_pred_LR4 = np.where(y_prob_LR4 > 0.5, 1, 0)

auc_roc4 = metrics.classification_report(y_test_LR4, y_pred_LR4)
false_rate4, true_rate4, thresholds4 = roc_curve(y_test_LR4, y_prob_LR4)
roc_auc4 = auc(false_rate4, true_rate4)

plt.subplot(1,2,2)
plt.title('ROC - GNB')
plt.plot(false_rate4, true_rate4, color='orange', label='AUC = %0.2f' % roc_auc4)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.axis('tight')
plt.ylabel('Truth Rate')
plt.xlabel('False  Rate')
plt.title('Very Hazard PM2.5')
plt.legend()








#####################################################Prediction Accuracy Plot##########################################



sheet_value = np.asarray(data)
sheet_value = sheet_value.astype(np.float)
for i in range(0,len(sheet_value[:,0])):                # devide the data value into different groups for predication
    if int(sheet_value[i, 0]) <= 12:
        sheet_value[i, 0] = int (1)
    if  int(sheet_value[i,0]) >12 and sheet_value[i,0]<35 :
        sheet_value[i,0] = int (2)
    if  int(sheet_value[i,0]) >=35 and sheet_value[i,0]<56 :
        sheet_value[i,0] = int (3)
    if  int(sheet_value[i,0]) >=56 and sheet_value[i,0]<150 :
        sheet_value[i,0] = int (4)
    if  int(sheet_value[i,0]) >=150 and sheet_value[i,0]<250 :
        sheet_value[i,0] = int (5)
    if  int(sheet_value[i,0]) >=250 and sheet_value[i,0]<350 :
        sheet_value[i,0] = int (6)
    if int(sheet_value[i, 0]) >= 350:
        sheet_value[i, 0] = int(7)


PM=sheet_value[:,0]
b=[]
fre_1=(PM == 1).sum()
b.append(fre_1)
fre_2=(PM == 2).sum()
b.append(fre_2)
fre_3=(PM == 3).sum()
b.append(fre_3)
fre_4=(PM == 4).sum()
b.append(fre_4)
fre_5=(PM == 5).sum()
b.append(fre_5)
fre_6=(PM == 6).sum()
b.append(fre_6)
fre_7=(PM == 7).sum()
b.append(fre_7)
a=[1,2,3,4,5,6,7]

plt.figure(5)
plt.bar(a[0],b[0],label='PM2.5 <12',color='r')
plt.bar(a[1],b[1],label='35>PM2.5 >12',color='b')
plt.bar(a[2],b[2],label='56>PM2.5 >35',color='y')
plt.bar(a[3],b[3],label='150>PM2.5 >56',color='pink')
plt.bar(a[4],b[4],label='250>PM2.5 >150',color='g')
plt.bar(a[5],b[5],label='300>PM2.5 >250',color='brown')
plt.bar(a[6],b[6],label='PM2.5 >300',color='orange')
plt.title('PM2.5 Concentration Occurancy/Frequency')
plt.legend()

# DEWP=sheet_value[:,1]
# TEMP=sheet_value[:,2]
# PRES=sheet_value[:,3]
# cdwd=sheet_value[:,4]
# lws=sheet_value[:,5]






ac2=accuracy_score(y_test_LR2, y_pred_LR2)
ac3=accuracy_score(y_test_LR3, y_pred_LR3)
ac4=accuracy_score(y_test_LR4, y_pred_LR4)
group=[1,2,3,4]

plt.figure(6)
plt.subplot(1,2,1)
plt.bar(group[0],ac1,label='Good for Outside Activity',color='r')
plt.bar(group[1],ac2,label='Unhealth or Unhealth for Sensitivity Group',color='black')
plt.bar(group[2],ac3,label='Not Recommended for Outside Activity',color='blue')
plt.bar(group[3],ac4,label='Harzard PM2.5 Air Quality',color='orange')
plt.title('Prediction Accuracy for Four Different Suggestion')
plt.ylim(0,1.5)
plt.legend()


plt.subplot(1,2,2)
auc=[roc_auc,roc_auc2,roc_auc3,roc_auc4]
b=[1,2,3,4]
print(auc)
plt.bar(b[0],auc[0],label='Logisc Regression ROC Accuracacy for Group 1',color='r')
plt.bar(b[1],auc[1],label='Logisc Regression ROC Accuracacy for Group 2',color='black')
plt.bar(b[2],auc[2],label='Logisc Regression ROC Accuracacy for Group 3',color='blue')
plt.bar(b[3],auc[3],label='Logisc Regression ROC Accuracacy for Group 4',color='orange')
plt.title('Prediction Accuracy for Four Different Suggestion')
plt.ylim(0,1.5)
plt.legend()
plt.show()

# http://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/