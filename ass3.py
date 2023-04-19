#Assignment 3
#Chunxu Fang
#cf146821

'''
• Read the ag.csv input file.  Expect that I will substitute a different file with the name ag.csv for testing.
• Identify the relevant columns and preprocess the text to make it suitable for further analysis.
• From the processed data, build three ML models:
• K-means clusters (with K equal to the number of classes in the data);
• a K-NN classifier;
• a Bayesian classifier.
• For each model compute and display evaluation metrics to help determine how well each model represents the data or provides predictive value for each class.
'''
##################################
print("Assignment 1: Chunxu Fang (cf146821)")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print("I use GitHub Copilot to help with the codes and annotations.\n")

# Read and check the data
address = '/Users/FANG/Documents/UAlbany/2023 Spring/Text Analysis/Assignment3/ag.csv'
df = pd.read_csv(address)
print("check the data\n")
print(df.head())
print("----------------------------------------------\n")

# Preprocess the text
print("----------------------------------------------\n")
print("Combine the title and description\n")
print("Remove extra whitespace within string\n")
print("Trim whitespace\n")
df["text"] = df["Title"] + " " + df["Description"]  # combine the title and description
df["text"] = df["text"].str.replace(r'\s+', ' ')  # remove extra whitespace within string
df["text"] = [s.strip() for s in df["text"]] #trim whitespace
print(df["text"][1:5,])
# Vectorize the text
print("----------------------------------------------\n")
print("Vectorize and scale the text\n")
##TF-IDF matrix
vectorizer = TfidfVectorizer(
    stop_words='english')
X = vectorizer.fit_transform(df['text'])
##Scale the matrix for K-mean, K-NN
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

# Split and get training and testing sets
print("----------------------------------------------\n")
print("Split and get training and testing sets\n")
y = df["Class Index"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# K-means 
print("----------------------------------------------\n")
print("K-means\n")
classifier_kmeans = KMeans(n_clusters=len(df["Class Index"].unique()),n_init='auto', random_state=2)
m_kmeans = classifier_kmeans.fit(X_train)
y_pred_kmeans = m_kmeans.predict(X_test)

# K-NN 
print("----------------------------------------------\n")
print("K-NN \n")
classifier_knn = KNeighborsClassifier(n_neighbors = 5)
m_knn = classifier_knn.fit(X_train, y_train)
y_pred_knn = m_knn.predict(X_test)

# naive bayes
print("----------------------------------------------\n")
print("naive bayes \n")
classifier_bayesian = MultinomialNB()
m_nb = classifier_bayesian.fit(X_train, y_train)
y_pred_nb = m_nb.predict(X_test)

# Evaluation metrics
print("----------------------------------------------\n")
print("Evaluation\n")
print("----------------------------------------------\n")
print("Overall Accuracy:")
acc_result = pd.DataFrame({
    "K-Means":[accuracy_score(y_test, y_pred_kmeans)],
    "K-NN":[accuracy_score(y_test, y_pred_knn)],
    "Navie-Bayesian":[accuracy_score(y_test, y_pred_nb)]
})
print(acc_result)
print("\n Detailed Results: \n")
print("K-means Clustering:")
print(confusion_matrix(y_test, y_pred_kmeans))
print(classification_report(y_test, y_pred_kmeans))

print("K-NN Classifier:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

print("Bayesian Classifier:")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))





