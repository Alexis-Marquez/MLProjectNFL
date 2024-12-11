from sklearn.preprocessing import LabelEncoder


def prepare_data(play_by_play_data, features):
    play_by_play_data = play_by_play_data.dropna(subset=['play_type'])

    play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'no_play']
    play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'kickoff']
    play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'extra_point']
    play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'qb_spike']
    play_by_play_data = play_by_play_data[play_by_play_data['play_type'] != 'qb_kneel']

    encoders = {}

    categorical_columns = [
        'game_half',
        'series_result', 'fixed_drive_result', 'play_type'
    ]

    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        play_by_play_data[column] = encoders[column].fit_transform(play_by_play_data[column])

    play_by_play_data = play_by_play_data.fillna(0)

    x = play_by_play_data[features]
    y = play_by_play_data['play_type']
    return x, y, encoders