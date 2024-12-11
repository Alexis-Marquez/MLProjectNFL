import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.functions import prepare_data

play_by_play_data = pd.read_csv("C:/Users/qewfh/PycharmProjects/MLProjectNFL/src/play_by_play_2024-copy.csv")

features = [
        'yardline_100', 'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining', 'game_half',
        'quarter_end', 'drive'
        , 'qtr', 'down', 'goal_to_go', 'ydstogo', 'yards_gained', 'timeout', 'posteam_timeouts_remaining'
        , 'defteam_timeouts_remaining', 'posteam_score', 'defteam_score', 'score_differential', 'no_score_prob',
        'opp_fg_prob'
        , 'opp_safety_prob', 'opp_td_prob', 'fg_prob', 'safety_prob', 'td_prob', 'series_success', 'series_result',
        'fixed_drive_result', 'drive_play_count'
        , 'drive_first_downs', 'drive_inside20', 'drive_ended_with_score', 'drive_quarter_start', 'drive_quarter_end',
        'out_of_bounds'
    ]

x, y, encoders = prepare_data(play_by_play_data, features)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

down_weight = 5
yard_weight = 15
goal_to_go_weight = 10
point_differential_weight = 5
x_train[:, features.index('down')] *= down_weight
x_test[:, features.index('down')] *= down_weight
x_train[:, features.index('yardline_100')] *= yard_weight
x_test[:, features.index('yardline_100')] *= yard_weight

x_train[:, features.index('ydstogo')] *= goal_to_go_weight
x_test[:, features.index('ydstogo')] *= goal_to_go_weight

x_test[:, features.index('score_differential')] *= point_differential_weight
x_train[:, features.index('score_differential')] *= point_differential_weight

logReg = LogisticRegression(max_iter=10000)

logReg.fit(x_train, y_train)

y_pred = logReg.predict(x_test)

coefficients = logReg.coef_[0]

importance = list(zip(features, np.abs(coefficients)))

importance.sort(key=lambda x: x[1], reverse=True)

for feature, coefficient in importance:
    print(f"Feature: {feature}, Coefficient: {coefficient}")

print(logReg.score(x_test, y_test))
print("accuracy: ", accuracy_score(y_test, y_pred))

print("classification report: \n", classification_report(y_test, y_pred))

classes = encoders['play_type'].classes_
print("Mapping of labels to encoded values: ")
for index, label in enumerate(classes):
    print(index, label)
print("\nSample Predictions (First 10):")
sample_df = pd.DataFrame({
    'Actual': y_test.iloc[:10],
    'Predicted': y_pred[:10]
    })
print(sample_df)

example_situation = dict.fromkeys(features, 0)
example_situation.update({
    'yardline_100':10,
    'quarter_seconds_remaining':20,
    'half_seconds_remaining':470,
    'game_seconds_remaining':1370,
    'game_half': 1,
    'quarter_end': 0,
    'drive': 5,
    'qtr': 1,
    'down': 4,
    'goal_to_go': 0,
    'ydstogo': 3,
    'posteam_timeouts_remaining': 2,
    'defteam_timeouts_remaining': 3,
    'posteam_score': 0,
    'defteam_score': 0,
    'score_differential': 0
})


def predict_play(model, situation):
    try:
        # Create a DataFrame with one row using the same feature names as training
        situation_df = pd.DataFrame([situation], columns=features)

        # Get prediction and probabilities
        situation_scaled = scaler.transform(situation_df)
        situation_scaled[:, features.index('down')] *= down_weight
        situation_scaled[:, features.index('yardline_100')] *= yard_weight
        situation_scaled[:, features.index('ydstogo')] *= goal_to_go_weight
        situation_scaled[:, features.index('score_differential')] *= point_differential_weight
        prediction = model.predict(situation_scaled)[0]
        probabilities = model.predict_proba(situation_scaled)[0]

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


def print_pred(model, situation):
    print("\nPredicting play for example situation...")
    predictions = predict_play(model, situation)
    if predictions:
        print("\nTop 3 likely plays:")
        for play_type, probability in predictions:
            print(f"{play_type}: {probability:.2%} confidence")


print_pred(logReg, example_situation)