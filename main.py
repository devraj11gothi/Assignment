# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# # Function to process a single file
# def process_file(file_path, model, le_transcript, le_field):
#     df = pd.read_csv(file_path, sep=',', header=None, names=[
#         'start_index', 'end_index', 'x_top_left', 'y_top_left', 
#         'x_bottom_right', 'y_bottom_right', 'transcript'
#     ])

#     def is_float(string):
#         try:
#             float(string)
#             return True
#         except ValueError:
#             return False
    
#     # Prepare features
#     df['is_numeric'] = df['transcript'].apply(lambda x: x.replace('.', '', 1).isdigit() if isinstance(x, str) else False)
#     df['numeric_value'] = df['transcript'].apply(lambda x: float(x) if is_float(x) else np.nan)
#     df['categorical_value'] = df['transcript'].apply(lambda x: x if not is_float(x) else np.nan)
    
#     # Encode categorical values
#     categorical_le = LabelEncoder()

#     df['categorical_encoded'] = categorical_le.fit_transform(df['categorical_value'].fillna('Unknown'))
    
#     # Prepare input for prediction
#     X = df[['start_index', 'end_index', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right', 
#             'is_numeric', 'numeric_value', 'categorical_encoded']]
    
#     # Make predictions
#     predictions = model.predict(X)
    
#     # Decode predictions
#     df['field'] = le_field.inverse_transform(predictions)
    
#     return df['start_index', 'end_index', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right', 'transcript', 'field']

# # Load and prepare training data
# folder_path = r"C:\Users\ADMIN\Desktop\Codes\Infrrd\dataset\dataset\train\boxes_transcripts_labels"
# tsv_files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
# df_list = []

# for tsv_file in tsv_files:
#     file_path = os.path.join(folder_path, tsv_file)
#     df = pd.read_csv(file_path, sep=',', header=None, names=[
#         'start_index', 'end_index', 'x_top_left', 'y_top_left', 
#         'x_bottom_right', 'y_bottom_right', 'transcript', 'field'
#     ]) 
#     df_list.append(df)

# df = pd.concat(df_list, ignore_index=True)

# def is_float(string):
#     try:
#         float(string)
#         return True
#     except ValueError:
#         return False

# # Prepare features
# df['is_numeric'] = df['transcript'].apply(lambda x: x.replace('.', '', 1).isdigit() if isinstance(x, str) else False)
# df['numeric_value'] = df['transcript'].apply(lambda x: float(x) if is_float(x) else np.nan)
# df['categorical_value'] = df['transcript'].apply(lambda x: x if not is_float(x) else np.nan)

# # LabelEncode the 'field' column
# le_field = LabelEncoder()
# df['field_encoded'] = le_field.fit_transform(df['field'])

# # LabelEncode the categorical values in 'transcript'
# le_transcript = LabelEncoder()
# df['categorical_encoded'] = le_transcript.fit_transform(df['categorical_value'].fillna('Unknown'))

# # Prepare features
# X = df[['start_index', 'end_index', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right', 
#         'is_numeric', 'numeric_value', 'categorical_encoded']]
# y = df['field_encoded']

# # Train the model
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X, y)

# # Process test files
# test_folder_path = r"C:\Users\ADMIN\Desktop\Codes\Infrrd\dataset\dataset\val\boxes_transcripts"
# output_folder_path = r"C:\Users\ADMIN\Desktop\Codes\Infrrd\predicted_outputs"

# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.tsv')]

# for test_file in test_files:
#     input_file_path = os.path.join(test_folder_path, test_file)
#     output_file_path = os.path.join(output_folder_path, f"{test_file}")
    
#     # Process the file and get predictions
#     df_predicted = process_file(input_file_path, rf_model, le_transcript, le_field)
    
#     # Save the results
#     df_predicted.to_csv(output_file_path, sep=',', index=False, header=False)

# print("Processing completed. Check the output folder for results.")




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

# Function to preprocess a single row
def preprocess_row(row, tfidf, numerical_features):
    # Combine non-numerical features into one text input for TF-IDF vectorization
    text = ' '.join([str(row[col]) for col in row.index if col not in numerical_features])
    text_features = tfidf.transform([text]).toarray()  # Vectorize text using TF-IDF
    numerical_values = [row[feat] for feat in numerical_features]  # Extract numerical values
    return np.concatenate([numerical_values, text_features[0]])  # Combine numerical and text features

# Load and preprocess the training data
print("Loading and preprocessing training data...")
train_data = pd.read_csv(r'C:\Users\ADMIN\Desktop\Codes\Infrrd\combined_data.csv')

# Create new features: box width and height from bounding box coordinates
train_data['box_width'] = train_data['x_bottom_right'] - train_data['x_top_left']
train_data['box_height'] = train_data['y_bottom_right'] - train_data['y_top_left']

# Clean and process the 'transcript' column
train_data['transcript'].fillna('', inplace=True)
train_data['transcript'] = train_data['transcript'].str.lower().str.strip()

# Combine all text columns into a single feature
numerical_features = ['start_index', 'end_index', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right', 'box_width', 'box_height']
text_columns = [col for col in train_data.columns if col not in numerical_features + ['field']]
train_data['combined_text'] = train_data[text_columns].fillna('').agg(' '.join, axis=1)

# Use TF-IDF to vectorize the text data
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf.fit(train_data['combined_text'])  # Fit on the combined text

# Prepare features and target variable
X = train_data.apply(lambda row: preprocess_row(row, tfidf, numerical_features), axis=1)
X = pd.DataFrame(X.tolist())  # Convert features into DataFrame

# Encode the 'field' target using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train_data['field'])

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Initialize RandomForest model with your grid search parameters
rf_model = RandomForestClassifier(
    n_estimators=120,  # Best from your grid search
    max_features=1.0,
    max_depth=None,
    max_samples=0.75,
    random_state=42,
    n_jobs=-1
)

# Initialize XGBoost model with the parameters provided earlier
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

# Combine RandomForest and XGBoost using VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'  # Soft voting to average probabilities
)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit the voting classifier
print("Training Voting Classifier (RandomForest + XGBoost)...")
voting_clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = voting_clf.predict(X_test)
print("Classification Report for Combined Model:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Function to preprocess the test data
def preprocess_test_data(test_data, tfidf, numerical_features):
    # Create the 'box_width' and 'box_height' features in the test data
    test_data['box_width'] = test_data['x_bottom_right'] - test_data['x_top_left']
    test_data['box_height'] = test_data['y_bottom_right'] - test_data['y_top_left']

    # Clean and process the 'transcript' column in the test data
    test_data['transcript'].fillna('', inplace=True)
    test_data['transcript'] = test_data['transcript'].str.lower().str.strip()

    # Combine all text columns into a single feature for TF-IDF
    text_columns = [col for col in test_data.columns if col not in numerical_features + ['field']]
    test_data['combined_text'] = test_data[text_columns].fillna('').agg(' '.join, axis=1)

    return test_data

# Process test files
test_folder = r'C:\Users\ADMIN\Desktop\Codes\Infrrd\claude\dataset\dataset\val\boxes_transcripts'
output_folder = r'C:\Users\ADMIN\Desktop\Codes\Infrrd\predicted_outputs'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("Processing test files...")
for filename in os.listdir(test_folder):
    if filename.endswith('.tsv'):
        input_path = os.path.join(test_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Read the test file
        test_data = pd.read_csv(input_path, sep=',')

        # Preprocess the test data (create box_width, box_height, and clean transcript)
        test_data = preprocess_test_data(test_data, tfidf, numerical_features)

        # Process each row and make predictions
        predictions = []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc=f"Processing {filename}"):
            features = preprocess_row(row, tfidf, numerical_features)  # Preprocess row
            pred = voting_clf.predict([features])[0]  # Predict field using VotingClassifier
            predictions.append(le.inverse_transform([pred])[0])  # Inverse transform to get original label
        
        # Add predictions to the test data
        test_data['predicted_field'] = predictions
        
        # Save the results
        test_data.to_csv(output_path, sep=',', index=False)

print("Processing complete. Results saved in output folder.")

