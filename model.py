# Import necessary libraries/packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,classification_report
import matplotlib.pyplot as plt
import joblib

# Data Ingestion
df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Separate X and Y features 
X = df["instruction"]
Y = df["intent"]

# Check the distribution of intents
print(Y.value_counts())

# Visualize the above distribution
distri = Y.value_counts()
distri.plot(kind='bar')
plt.show()

# Divide the data into training and testing to evaluate model performance
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.8,random_state=21)

# create a dictionary that contains base models for all algorithms
dct = {
    "Logistic" : LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier(),
    "SVM":SVC()
}

# Create a function to evaluate model metrics

def algorithm_evaluation(model):
    # Pipleline for Text Preprocessing and model building
    pipeline = Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("model",model)])
    # Fit the model on X and Y features
    pipeline.fit(xtrain,ytrain)
    # Predict the results
    ypred = pipeline.predict(xtest)
    # Evaluate the model
    f1_test = f1_score(ytest,ypred,average='macro')
    return f1_test

# create empty lists to store the results
test_scores = []

for name,model in dct.items():
    test = algorithm_evaluation(model)
    # store the results in above created lists
    test_scores.append(test)
    
# create a dataframe to show the above results
res = {
    "Name":list(dct.keys()),
    "F1 Test Scores":test_scores
}
df_res = pd.DataFrame(res)
model_highf1score = df_res.nlargest(1,columns="F1 Test Scores")[["Name"]]
print(f"{model_highf1score} Model is providing highest f1-score. Considering this model for final predictions")

# Build the model 
tfidf = TfidfVectorizer() # Text preprocessing
xtrain_new = tfidf.fit_transform(xtrain)

model_svc = SVC()

# Fit the model
model_svc.fit(xtrain_new,ytrain)

# Evaluate the model

xtest_new = tfidf.transform(xtest)
ypreds = model.predict(xtest_new)
f1 = f1_score(ytest,ypreds,average='macro')
print(classification_report(ytest,ypreds))

# save the model and vectorizer
joblib.dump(tfidf,"tfidfvectorizer.pkl")
joblib.dump(model_svc,"chatbot__nlpmodel.pkl")