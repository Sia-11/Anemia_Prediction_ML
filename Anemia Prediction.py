import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv("C:/Users/Imthiyaz/Downloads/archive (1)/anemia.csv")
print(dataset)
print(f"Shape of dataset -> {dataset.shape}")
print(dataset.head(10))

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

LR_c = LogisticRegression()
Knn_c = KNeighborsClassifier(n_neighbors=4)
SVM_c = SVC(kernel="linear", C=1)



models=[LR_c,Knn_c,SVM_c]

accuracy_scores={}

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_name = model.__class__.__name__
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy
    print(model,"Accuracy : ",accuracy)
    print(model,"Confusion Matrix-")
    print(confusion_matrix(y_test,y_pred))
    print()
    
def anemia_predictor(Gender,Hemoglobin, MCH, MHCH,MCV ):
    a = SVM_c.predict(sc.transform([[Gender,Hemoglobin, MCH, MHCH,MCV]]))
    if a == 0:
        print("You Don't have anemia.")
    else:
        print("You have anemia ...seek doctor's advice.")

    
Gender = input("Gender[0-Male / 1-Female]: ")
Hemoglobin = input("Hemoglobin:")
MCH = input("MCH:")
MHCH = input("MHCH:")
MCV = input("MCV:")
anemia_predictor(Gender,Hemoglobin, MCH, MHCH,MCV )


plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison - Accuracy Scores')

plt.ylim([0, 1])
plt.xticks(rotation=45)

best_model = max(accuracy_scores, key=accuracy_scores.get)
plt.bar(best_model, accuracy_scores[best_model], color='green')

plt.tight_layout()
plt.show()

print(f"Best Fitting Model = '{best_model}' with an accuracy of {accuracy_scores[best_model]:.2f}")




