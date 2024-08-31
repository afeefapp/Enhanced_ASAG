from sentence_transformers import SentenceTransformer
import pandas as pd
import decimal
from decimal import Decimal, getcontext
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler




#Loading the finetuned sentence transformer lodel
model=SentenceTransformer("/ASAG/trained_model_mistral_50/content/trained_model")

#Training set Preprocessing
data = pd.read_csv("/ASAG/ASAG_train_new.csv")
data = data[['id','Question','RefAns', 'StdAns', 'Score']]

getcontext().rounding = decimal.ROUND_HALF_UP
getcontext().prec=1
label=[]
for i in range(len(data)):
  lb = int(round(Decimal(data['Score'].iloc[i]),0))
  if(lb==0 or lb==1):
    sc = 0
  elif(lb==2 or lb==3):
    sc = 1
  else:
    sc = 2
  label.append(sc)

data.insert(5,"label", label, True)


#Test set preprocessing
test = pd.read_csv("/ASAG/ASAG_test.csv")
test = test[['Question','RefAns', 'StdAns', 'Score']]
getcontext().rounding = decimal.ROUND_HALF_UP
getcontext().prec=1
label=[]
for i in range(len(test)):
  lb = int(round(Decimal(test['Score'].iloc[i]),0))
  if(lb==0 or lb==1):
    sc = 0
  elif(lb==2 or lb==3):
    sc = 1
  else:
    sc = 2
  label.append(sc)

test.insert(4,"label", label, True)

test['emb1'] = test['StdAns'].astype(str).apply(model.encode)
test['emb2'] = test['RefAns'].astype(str).apply(model.encode)
test['embeddings']=abs(test['emb1']-test['emb2'])
X_test = test['embeddings'].to_list()
y_test = test['label'].to_list()

#Dataset Class distribution
count=data["label"].value_counts(ascending=True)
count.plot.barh()
plt.title("Frequency of Classes")
plt.show()

a=[]
for i in range(3):
  a.append(max(count)-count[i])

#Load augmented dataset
df = pd.read_csv("/ASAG/ASAG_train_mistral_unique.csv")
aug = df.loc[27413:] # 27413 is the size of original ASAG dataset
aug = aug[['id','Question','RefAns', 'StdAns', 'Score']]

# Do the same process of label creation on the augmented entries.
label=[]
for i in range(len(aug)):
  lb = int(round(Decimal(aug['Score'].iloc[i]),0))
  if(lb==0 or lb==1):
    sc = 0
  elif(lb==2 or lb==3):
    sc = 1
  else:
    sc = 2
  label.append(sc)

aug.insert(5,"label", label, True)
train=data

#Oversampling using augmented instances
rslt0= aug.loc[aug["label"]==0]
rslt1= aug.loc[aug["label"]==1]
rslt2= aug.loc[aug["label"]==2]

b=[]
alpha = 0.5 # 'alpha' is the hyperparameter that can be tuned to get the best result
for i in range(len(a)):
  if(count[i]<(max(count)*alpha)):
    b.append((max(count))*alpha - count[i])
  else:
    b.append(0)

if(len(rslt0)<b[0]):
  fraction0 = 1
else:
  fraction0 = b[0]/len(rslt0)

if(len(rslt1)<b[1]):
  fraction1 = 1
else:
  fraction1 = b[1]/len(rslt1)

if(len(rslt2)<b[2]):
  fraction2 = 1
else:
  fraction2 = b[2]/len(rslt2)

rslt0 = rslt0.groupby("id").apply(lambda g: g.sample(frac=fraction0,random_state=12))
rslt1 = rslt1.groupby("id").apply(lambda g: g.sample(frac=fraction1,random_state=12))
rslt2 = rslt2.groupby("id").apply(lambda g: g.sample(frac=fraction2,random_state=12))

train = pd.concat([data,rslt0,rslt1,rslt2],axis=0)

# Preprocessing of oversampled training set
train['emb1'] = train['StdAns'].astype(str).apply(model.encode)
train['emb2'] = train['RefAns'].astype(str).apply(model.encode)

train['embeddings']=abs(train['emb1']-train['emb2'])

train= train[["embeddings", "label"]]
X = train['embeddings'].to_list()
y = train['label'].to_list()

# Use any one of the following synthetic oversampling technique before training the classifier
# SMOTE Over-sampling
ros = SMOTE(random_state=42)
X, y = ros.fit_resample(X, y)
print("\nClass Distribution After Random Over-sampling:")
print(pd.Series(y).value_counts())

'''
# ADASYN Over-sampling
ros = ADASYN(random_state=42, n_neighbors=1)
X,y= ros.fit_resample(X, y)
print("\nClass Distribution After Random Over-sampling:")
print(pd.Series(y).value_counts())

#Random Over-sampling
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
print("\nClass Distribution After Random Over-sampling:")
print(pd.Series(y).value_counts())
'''

#Train SVM classifier
svc = SVC()
svc.fit(X, y)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
f1_svc = round(f1_score(y_test, y_pred, average='macro')*100,2)
print("Accuracy and F1-score of SVM:")
print(acc_svc, f1_svc)

#Train KNN classifier
knn = KNeighborsClassifier()
knn.fit(X,y)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)
f1_knn = round(f1_score(y_test, y_pred, average='macro')*100,2)
print("Accuracy and F1-score of KNN:")
print(acc_knn, f1_knn)

#Train XGBoost classifier
xgbt =  xgb.XGBClassifier(n_jobs=-1)
xgbt.fit(X,y)
y_pred = xgbt.predict(X_test)
acc_xgbt = round(accuracy_score(y_pred, y_test) * 100, 2)
f1_xgbt = round(f1_score(y_test, y_pred, average='macro')*100,2)
print("Accuracy and F1-score of XGBoost:")
print(acc_xgbt, f1_xgbt)


