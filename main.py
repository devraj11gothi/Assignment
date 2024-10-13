import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def preprocess_row(row, tfidf, numerical_features):
    text = ' '.join([str(row[col]) for col in row.index if col not in numerical_features])
    text_features = tfidf.transform([text]).toarray() 
    numerical_values = [row[feat] for feat in numerical_features]
    return np.concatenate([numerical_values, text_features[0]])

print("Loading and preprocessing training data...")
train_data = pd.read_csv(r'C:\Users\ADMIN\Desktop\Codes\Infrrd\combined_data.csv')

train_data['box_width'] = train_data['x_bottom_right'] - train_data['x_top_left']
train_data['box_height'] = train_data['y_bottom_right'] - train_data['y_top_left']

train_data['transcript'].fillna('', inplace=True)
train_data['transcript'] = train_data['transcript'].str.lower().str.strip()

numerical_features = ['start_index', 'end_index', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right', 'box_width', 'box_height']
text_columns = [col for col in train_data.columns if col not in numerical_features + ['field']]
train_data['combined_text'] = train_data[text_columns].fillna('').agg(' '.join, axis=1)

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf.fit(train_data['combined_text']) 

X = train_data.apply(lambda row: preprocess_row(row, tfidf, numerical_features), axis=1)
X = pd.DataFrame(X.tolist())

le = LabelEncoder()
y = le.fit_transform(train_data['field'])

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

rf_model = RandomForestClassifier(
    n_estimators=120, 
    max_features=1.0,
    max_depth=None,
    max_samples=0.75,
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    random_state=42,
    n_jobs = -1
)

voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'  
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("Training Voting Classifier (RandomForest + XGBoost)...")
voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)
print("Classification Report for Combined Model:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

def preprocess_test_data(test_data, tfidf, numerical_features):
    test_data['box_width'] = test_data['x_bottom_right'] - test_data['x_top_left']
    test_data['box_height'] = test_data['y_bottom_right'] - test_data['y_top_left']

    test_data['transcript'].fillna('', inplace=True)
    test_data['transcript'] = test_data['transcript'].str.lower().str.strip()

    text_columns = [col for col in test_data.columns if col not in numerical_features + ['field']]
    test_data['combined_text'] = test_data[text_columns].fillna('').agg(' '.join, axis=1)

    return test_data

test_folder = r'C:\Users\ADMIN\Desktop\Codes\Infrrd\claude\dataset\dataset\val\boxes_transcripts'
output_folder = r'C:\Users\ADMIN\Desktop\Codes\Infrrd\predicted_outputs'

os.makedirs(output_folder, exist_ok=True)

print("Processing test files...")
for filename in os.listdir(test_folder):
    if filename.endswith('.tsv'):
        input_path = os.path.join(test_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        test_data = pd.read_csv(input_path, sep=',')

        test_data = preprocess_test_data(test_data, tfidf, numerical_features)

        predictions = []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc=f"Processing {filename}"):
            features = preprocess_row(row, tfidf, numerical_features) 
            pred = voting_clf.predict([features])[0]  
            predictions.append(le.inverse_transform([pred])[0])  
        
        test_data['predicted_field'] = predictions
        
        test_data.to_csv(output_path, sep=',', index=False)

print("Processing complete. Results saved in output folder.")
