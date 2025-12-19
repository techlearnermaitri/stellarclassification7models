import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score

df=pd.read_csv('star_classification.csv')
print(df)

print(df.shape) #we have 100000 objects and 17 features

df.isnull().sum()

df = df.dropna()
print(df)

df = df.drop(columns=[
    'obj_ID', 'run_ID', 'rerun_ID', 'cam_col',
    'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID'])
print(df)

df['class'].unique()

df['class'].value_counts()
sns.countplot(x = df['class'])
plt.show()

df['class']=df['class'].map({'GALAXY':1,'STAR':0,'QSO':2})
print(df)

X = df[['u', 'g', 'r', 'i', 'z', 'redshift']]
print(X)

y = df['class']
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train svm instead of resampling because its better to use bootstrap methods instead
model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print(" Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))


#logostic regression

print(" Logistic Regression ")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))


#knn

print(" KNN ")

knn_model = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance'
)

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))


#DT
print("Decision Tree ")

dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=None
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))


#GNB

print(" Gaussian Naive Bayes ")

gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
gnb_pred = gnb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, gnb_pred))
print("Classification Report:\n", classification_report(y_test, gnb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, gnb_pred))


#RF

print(" Random Forest ")

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))


from xgboost import XGBClassifier

print(" XGBoost ")

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

#LASSO AND RIDGE FOR CV
#RIDGE
print("Lasso Logistic Regression (L1) ")

lasso_model = LogisticRegression(
    penalty='l1',
    C=0.5,                 # stronger regularization
    solver='liblinear',    # required for L1
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, lasso_pred))
print("Classification Report:\n", classification_report(y_test, lasso_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lasso_pred))


#RIDGE

print("Ridge Logistic Regression (L2) ")

ridge_model = LogisticRegression(
    penalty='l2',
    C=1.0,                 # smaller C → stronger regularization
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, ridge_pred))
print("Classification Report:\n", classification_report(y_test, ridge_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ridge_pred))



kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

#logistic regression cv
lr_kf = cross_val_score(
    lr_model, X, y,
    cv=kf,
    scoring='accuracy'
)

print("\nLogistic Regression K-Fold Accuracy:")
print("Mean:", lr_kf.mean())
print("Std :", lr_kf.std())

#knn cv

knn_kf = cross_val_score(
    knn_model, X, y,
    cv=kf,
    scoring='accuracy'
)

print("\nKNN K-Fold Accuracy:")
print("Mean:", knn_kf.mean())
print("Std :", knn_kf.std())

#decision tree cv
dt_kf = cross_val_score(
    dt_model, X, y,
    cv=kf,
    scoring='accuracy'
)

print("\nDecision Tree K-Fold Accuracy:")
print("Mean:", dt_kf.mean())
print("Std :", dt_kf.std())

#gaussian NB cv
gnb_kf = cross_val_score(
    gnb_model, X, y,
    cv=kf,
    scoring='accuracy'
)

print("\nGaussian NB K-Fold Accuracy:")
print("Mean:", gnb_kf.mean())
print("Std :", gnb_kf.std())

#Random Forets v
rf_kf = cross_val_score(
    rf_model, X, y,
    cv=kf,
    scoring='accuracy'
)

print("\nRandom Forest K-Fold Accuracy:")
print("Mean:", rf_kf.mean())
print("Std :", rf_kf.std())



#xgboost cv
xgb_kf = cross_val_score(
    xgb_model, X, y,
    cv=kf,
    scoring='accuracy'
)

print("\nXGBoost K-Fold Accuracy:")
print("Mean:", xgb_kf.mean())
print("Std :", xgb_kf.std())

kf_results = {
    "Logistic Regression": lr_kf.mean(),
    "KNN": knn_kf.mean(),
    "Decision Tree": dt_kf.mean(),
    "Gaussian NB": gnb_kf.mean(),
    "Random Forest": rf_kf.mean(),
    "XGBoost": xgb_kf.mean()
}

print(" K-Fold Cross-Validation Results ")
print("Logistic Regression:", lr_kf.mean())
print("KNN:", knn_kf.mean())
print("Decision Tree:", dt_kf.mean())
print("Gaussian NB:", gnb_kf.mean())
print("Random Forest:", rf_kf.mean())
print("XGBoost:", xgb_kf.mean())


