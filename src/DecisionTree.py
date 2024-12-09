import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import customFunctions

play_by_play_data = pd.read_csv("play_by_play_2024.csv")

play_by_play_data = play_by_play_data.dropna(subset=['play_type'])

play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'no_play']
play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'kickoff']
play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'extra_point']


play_type_classes = play_by_play_data['play_type'].unique()

encoders = {}


categorical_columns = [
    'game_half',
    'series_result', 'fixed_drive_result', 'play_type'
]

for column in categorical_columns:
    encoders[column] = LabelEncoder()
    play_by_play_data[column] = encoders[column].fit_transform(play_by_play_data[column])

play_by_play_data = play_by_play_data.fillna(0)

features = [
    'yardline_100','quarter_seconds_remaining','half_seconds_remaining','game_seconds_remaining', 'game_half','quarter_end','drive'
    ,'qtr','down','goal_to_go','ydstogo','yards_gained','timeout','posteam_timeouts_remaining'
    ,'defteam_timeouts_remaining','posteam_score','defteam_score','score_differential','no_score_prob','opp_fg_prob'
    ,'opp_safety_prob','opp_td_prob','fg_prob','safety_prob','td_prob','series_success','series_result','fixed_drive_result','drive_play_count'
    ,'drive_first_downs','drive_inside20','drive_ended_with_score','drive_quarter_start','drive_quarter_end','out_of_bounds'
]

X = play_by_play_data[features]
y = play_by_play_data['play_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

tree = DecisionTreeClassifier(max_depth=15, min_samples_split=50, min_samples_leaf=10, random_state=20)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = np.mean(y_test == y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")
print("\nSample Predictions (First 10):")
sample_df = pd.DataFrame({
    'Actual': y_test.iloc[:10],
    'Predicted': y_pred[:10]
    })
print(sample_df)

# def evaluate_model(model, X_test, y_test):
#     """Evaluate the model and print various metrics"""
#     y_pred = model.predict(X_test)
#
#     print("\nClassification Report:")
#     # Add zero_division parameter to handle classes with no predictions
#     print(classification_report(y_test, y_pred, zero_division=0))
#
#     # Add value counts to see class distribution
#     print("\nClass Distribution in Test Set:")
#     print(pd.Series(y_test).value_counts())
#
#     print("\nPredicted Class Distribution:")
#     print(pd.Series(y_pred).value_counts())
#
#     plt.figure(figsize=(12, 8))
#     cm = confusion_matrix(y_test, y_pred)
#     # Create simple numeric labels for the confusion matrix
#     labels = sorted(list(set(y_test)))
#     sns.heatmap(cm, annot=True, fmt='d',
#                 xticklabels=labels,
#                 yticklabels=labels)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
#
#     # Calculate and print overall accuracy
#     accuracy = np.mean(y_test == y_pred)
#     print(f"\nOverall Accuracy: {accuracy:.2%}")
#
#     # Print sample predictions
#     print("\nSample Predictions (First 10):")
#     sample_df = pd.DataFrame({
#         'Actual': y_test.iloc[:10],
#         'Predicted': y_pred[:10]
#     })
#     print(sample_df)
#
#
# def plot_feature_importance(model, feature_names):
#     """Plot feature importance"""
#     importances = pd.DataFrame({
#         'feature': feature_names,
#         'importance': model.feature_importances_
#     })
#     importances = importances.sort_values('importance', ascending=False)
#
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='importance', y='feature', data=importances)
#     plt.title('Feature Importance')
#     plt.tight_layout()
#     plt.show()
#
#     return importances


example_situation = dict.fromkeys(features, 0)
example_situation.update({
    'yardline_100':20,
    'quarter_seconds_remaining':20,
    'half_seconds_remaining':470,
    'game_seconds_remaining':1370,
    'game_half': 1,
    'quarter_end': 0,
    'drive': 5,
    'qtr': 1,
    'down': 3,
    'goal_to_go': 0,
    'ydstogo': 5,
    'posteam_timeouts_remaining': 2,
    'defteam_timeouts_remaining': 3,
    'posteam_score': 21,
    'defteam_score': 0,
    'score_differential': 21
})

def predict_play(model, situation):
    """Predict the play type for a given situation"""
    try:
        # Create a DataFrame with one row using the same feature names as training
        situation_df = pd.DataFrame([situation], columns=features)

        # Get prediction and probabilities
        prediction = model.predict(situation_df)[0]
        probabilities = model.predict_proba(situation_df)[0]

        # Get top 3 most likely plays
        top_plays = []
        for idx in np.argsort(probabilities)[-3:][::-1]:
            play = idx
            probability = probabilities[idx]
            play_name = encoders['play_type'].inverse_transform([play])[0]
            top_plays.append((play_name, probability))

        return top_plays
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return []

def print_pred(model, example_situation):
    print("\nPredicting play for example situation...")
    predictions = predict_play(model, example_situation)
    if predictions:
        print("\nTop 3 likely plays:")
        for play_type, probability in predictions:
            print(f"{play_type}: {probability:.2%} confidence")

print_pred(tree, example_situation)
