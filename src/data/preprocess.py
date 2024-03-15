import pandas as pd
import pickle

def cumulative_concat_reverse(group):
    """
    Create context windows by reversing and cumulatively concatenating utterances.
    """
    reversed_utts_with_sep = (' \n ' + group['input']).iloc[::-1].cumsum()
    return reversed_utts_with_sep.iloc[::-1].str.lstrip('[SEP]')

def create_context(data):
    """
    Enriches the data with dialogue context.
    """
    data['input'] = "Speaker - " + data['Speaker'] + ": " + data['Utterance']
    data = data.sort_values(by = ['Season', 'Episode', 'Dialogue_ID', 'Utterance_ID'], ascending = [True, True, True, False])
    data['Context'] = data.groupby(['Season', 'Episode', 'Dialogue_ID'], group_keys = False).apply(cumulative_concat_reverse)
    data = data.sort_values(by = ['Season', 'Episode', 'Dialogue_ID', 'Utterance_ID'])
    data['Context'] = data['Context'].str.lstrip('\n ')
    return data

def group_speakers(data, speakers_to_keep):
    """
    Create "Other" category for infrequent speakers.
    """
    data['Speaker'] = data['Speaker'].data(data['Speaker'].isin(speakers_to_keep), 'Other')
    return data

def load_text_data(file_paths):
    """
    Load and preprocess text data from specified file paths.
    """
    data_frames = [pd.read_csv(file_path) for file_path in file_paths]
    processed_data_frames = [create_context(group_non_frequent_characters(df)) for df in data_frames]
    return processed_data_frames

def load_image_data(image_data_paths):
    """
    Loads image data from specified pickle files.
    """
    image_data = [pickle.load(open(file_path, 'rb')) for file_path in image_data_paths]
    return image_data

def attach_image_data(text_data_frames, image_data):
    """
    Attaches image features to text data frames based on 'new_id'.
    """
    for df, img_data in zip(text_data_frames, image_data):
        
        # Remove non-matchable rows
        df['new_id'] = df['Dialogue_ID'].astype(str) + "_" + df['Utterance_ID'].astype(str)
        df['label'] = df['new_id'].apply(lambda x: img_data.get(x, {}).get('label', 'DNE'))
        df = df[df['label'] != 'DNE']

        # Create image frames
        df['first_frame'] = df['new_id'].apply(lambda x: img_data.get(x, {}).get('video_features')[0])
        df['middle_frame'] = df['new_id'].apply(lambda x: img_data.get(x, {}).get('video_features')[(img_data.get(x, {}).get('video_features').shape[0]) // 2])
        df['last_frame'] = df['new_id'].apply(lambda x: img_data.get(x, {}).get('video_features')[-1])
                
    return text_data_frames

def save_preprocessed_data(data_frames, save_paths):
    """
    Saves preprocessed data frames to pickle files.
    """
    for df, path in zip(data_frames, save_paths):
        with open(path, 'wb') as file:
            pickle.dump(df, file)