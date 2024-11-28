import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

play_by_play_data = pd.read_csv("C:/Users/qewfh/PycharmProjects/MLProjectNFL/src/play_by_play_2024.csv")

label_encoder = LabelEncoder()

play_by_play_data['game_half'] = label_encoder.fit_transform(play_by_play_data['game_half'])
play_by_play_data['pass_location'] = label_encoder.fit_transform(play_by_play_data['pass_location'])
play_by_play_data['run_location'] = label_encoder.fit_transform(play_by_play_data['run_location'])
play_by_play_data['run_gap'] = label_encoder.fit_transform(play_by_play_data['run_gap'])
play_by_play_data['series_result'] = label_encoder.fit_transform(play_by_play_data['series_result'])
play_by_play_data['play_type_nfl'] = label_encoder.fit_transform(play_by_play_data['play_type_nfl'])
play_by_play_data['fixed_drive_result'] = label_encoder.fit_transform(play_by_play_data['fixed_drive_result'])
play_by_play_data['drive_start_transition'] = label_encoder.fit_transform(play_by_play_data['drive_start_transition'])
play_by_play_data['drive_end_transition'] = label_encoder.fit_transform(play_by_play_data['drive_end_transition'])
play_by_play_data['play_type'] = label_encoder.fit_transform(play_by_play_data['play_type'])

play_by_play_data = play_by_play_data.fillna(0)

print(play_by_play_data.columns[play_by_play_data.isnull().any()])

x = play_by_play_data.drop(columns=['play_type'])

y = play_by_play_data['play_type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# custom_plays = pd.read_csv('C:/Users/qewfh/PycharmProjects/MLProjectNFL/src/customPlays.csv')
# label_encoder1 = LabelEncoder()
# custom_plays['game_half'] = label_encoder1.fit_transform(custom_plays['game_half'])
# custom_plays['pass_location'] = label_encoder1.fit_transform(custom_plays['pass_location'])
# custom_plays['run_location'] = label_encoder1.fit_transform(custom_plays['run_location'])
# custom_plays['run_gap'] = label_encoder1.fit_transform(custom_plays['run_gap'])
# custom_plays['series_result'] = label_encoder1.fit_transform(custom_plays['series_result'])
# custom_plays['play_type_nfl'] = label_encoder1.fit_transform(custom_plays['play_type_nfl'])
# custom_plays['fixed_drive_result'] = label_encoder1.fit_transform(custom_plays['fixed_drive_result'])
# custom_plays['drive_start_transition'] = label_encoder1.fit_transform(custom_plays['drive_start_transition'])
# custom_plays['drive_end_transition'] = label_encoder1.fit_transform(custom_plays['drive_end_transition'])
# custom_plays['play_type'] = label_encoder1.fit_transform(custom_plays['play_type'])

# custom_plays = custom_plays.fillna(0)
#
# y_test = custom_plays['play_type']
# x_test = custom_plays.drop(columns=['play_type'])

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
kmeans = KNeighborsClassifier(n_neighbors=30, metric='euclidean')
kmeans.fit(x_train, y_train)

y_pred = kmeans.predict(x_test)
print(kmeans.score(x_test, y_test))
print("accuracy: ", accuracy_score(y_test, y_pred))
print(y_pred)
classes = label_encoder.classes_
print("Mapping of labels to encoded values: ")
for index, label in enumerate(classes):
    print(index, label)