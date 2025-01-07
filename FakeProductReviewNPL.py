# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('Dataset/fake_reviews_dataset.csv')

# Dataset information
print("\n",df.info())
print("\n",df.describe())
print("\n",df.head())

# Define features (X) and target (y)
X = df['text_']
y = df['label']

# Split the dataset into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

# Initialize the TfidfVectorizer to convert test into numerical features
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression classifier
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n Accuracy : {accuracy}")
print(f"\n Classification Report : {report}")
