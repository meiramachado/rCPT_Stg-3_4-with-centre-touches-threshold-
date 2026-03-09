import os
import pandas as pd
import numpy as np
import re

def apply_formulas(df):
    binary_df = pd.DataFrame()
    binary_df['Time'] = df.iloc[:, 0]  # Time column

    event_mappings = {
        'TTL #1': 'TTL #1',
        'Hit': 'Hit',
        'Missed Hit': 'Missed Hit',
        'Correct Rejection': 'Correct Rejection',
        'Non Correction Trial Miskake': 'False Alarm',
        'Reward Collected Start ITI': 'Reward Collected Start ITI',
        'Feeder #1': 'Reward Delivery',
        'Correction Trial Mistake': 'Correction Trial Mistake',
        'Correction Trial Correct Rejection': 'Correction Trial Correct Rejection',
        'Centre Screen Touches': 'Centre Screen Touches'
    }

    # Identify Display Image rows
    is_display_image = (
        (df.iloc[:, 2] == "Condition Event") &
        (df.iloc[:, 3] == "Display Image") &
        (df.iloc[:, 5] == 4)
    )
    binary_df['Display Image Binary'] = is_display_image.astype(int)
    display_indices = df[is_display_image].index.to_list()

    # Add binary columns for all events except "Reward Delivery" (custom logic)
    for event_value, col_name in event_mappings.items():
        if col_name != "Reward Delivery":
            binary_df[col_name] = np.where(df.iloc[:, 3] == event_value, 1, 0)

    # Reward Delivery: only if Hit occurred before in the same trial
    reward_delivery = np.zeros(len(df))
    for i, disp_idx in enumerate(display_indices):
        next_disp = display_indices[i + 1] if i + 1 < len(display_indices) else len(df)
        trial = df.iloc[disp_idx + 1 : next_disp]
        has_hit = 'Hit' in trial.iloc[:, 3].values
        feeder_indices = trial[trial.iloc[:, 3] == 'Feeder #1'].index
        if has_hit and not feeder_indices.empty:
            for idx in feeder_indices:
                reward_delivery[idx] = 1
    binary_df["Reward Delivery"] = reward_delivery

    # Raw Centre Screen Touches
    binary_df["Centre Screen Touches"] = np.where(df.iloc[:, 3] == 'Centre Screen Touches', 1, 0)

    # Filtered Centre Touches: only those >1s apart
    times = df.iloc[:, 0].values
    centre_indices = df[df.iloc[:, 3] == 'Centre Screen Touches'].index.to_list()
    filtered_centre = np.zeros(len(df))

    if centre_indices:
        previous_time = -np.inf
        for idx in centre_indices:
            current_time = times[idx]
            if current_time - previous_time > 1:  # 1s threshold
                filtered_centre[idx] = 1
                previous_time = current_time

    binary_df["Centre Touches (>1s apart)"] = filtered_centre

    # Outcome alignment to Display Image
    label_map = {
        "Hit": "Hit",
        "Missed Hit": "Missed Hit",
        "Correct Rejection": "Correct Rejection",
        "False Alarm": "Non Correction Trial Miskake",
        "Correction Trial Mistake": "Correction Trial Mistake",
        "Correction Trial Correct Rejection": "Correction Trial Correct Rejection"
    }
    is_outcome = (
        (df.iloc[:, 2] == "Condition Event") &
        (df.iloc[:, 5].between(5, 6))
    )
    outcome_df = df[is_outcome]
    for label, expected_name in label_map.items():
        aligned = np.zeros(len(df))
        for i, disp_idx in enumerate(display_indices):
            next_disp = display_indices[i + 1] if i + 1 < len(display_indices) else len(df)
            trial_outcomes = outcome_df[(outcome_df.index > disp_idx) & (outcome_df.index < next_disp)]
            if any(trial_outcomes.iloc[:, 3] == expected_name):
                aligned[disp_idx] = 1
        binary_df[f'{label} (with Display)'] = aligned

    return binary_df


def generate_binary_file(input_folder, output_folder, filename_prefix="TCN"):
    input_folder = os.path.normpath(input_folder)
    output_folder = os.path.normpath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.startswith(filename_prefix) and file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)

            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                data_start_index = None
                animal_id = None
                date_time = None

                for i, line in enumerate(lines):
                    if "Evnt_Time" in line:
                        data_start_index = i
                        break
                    elif "Animal ID" in line:
                        animal_id = line.strip().split(",")[1].strip()
                    elif "Date/Time" in line:
                        date_time = line.strip().split(",")[1].strip()

                if data_start_index is None:
                    print(f"Skipping file {file}: 'Evnt_Time' column not found.")
                    continue

                df = pd.read_csv(input_path, skiprows=data_start_index, encoding='utf-8')
                binary_df = apply_formulas(df)

                if animal_id and date_time:
                    safe_date_time = re.sub(r'[: /]', '_', date_time)
                    output_filename = f"{animal_id}_{safe_date_time}.csv"
                else:
                    output_filename = f"Binary_{file}"

                output_path = os.path.join(output_folder, output_filename)
                binary_df.to_csv(output_path, index=False)
                print(f"Successfully processed: {file} -> {output_filename}")

            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue


if __name__ == "__main__":
    # <<< UPDATE THESE PATHS BEFORE RUNNING >>>
    input_path = "/Volumes/MEIRA_DATA/CPT/Ketamine_exp/Ketamine Probes/Abet Raw files"
    output_path = "/Volumes/MEIRA_DATA/CPT/Ketamine_exp/Ketamine Probes"

    generate_binary_file(input_folder=input_path, output_folder=output_path)
