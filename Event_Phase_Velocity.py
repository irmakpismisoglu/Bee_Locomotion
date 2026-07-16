"""
Event-phase velocity analysis for honeybee dyads.

This script calculates the mean velocity of each bee before and after two
manually annotated behavioural events:

    1. Crossing
    2. Contact

The analysis uses filtered DeepLabCut tracking coordinates from the abdomen
body part. Pixel displacement is converted to millimetres using the measured
Petri dish diameter for each dyad.

For every bee, the script exports:

    - Crossing timestamp
    - Contact timestamp
    - Mean velocity before crossing
    - Mean velocity after crossing
    - Mean velocity before contact
    - Mean velocity after contact
    - Corresponding event frames

Velocity is calculated between consecutive video frames:

    velocity = displacement_in_mm × frames_per_second

The output contains one row per bee.

Author: Irmak Pismisoglu
Project: MicroBeeH
"""

import os

import numpy as np
import pandas as pd


# Paths

TRACKING_FOLDER = r"your tracking data.xlsx or. csv"

PETRI_SUMMARY_FILE = (
    r"\bee_velocity_distance_summary.csv")

EVENT_FILE = r"I:your file with event times that you want to calculate your before/after velocity"

OUTPUT_EXCEL = r"your uoutput file.xlsx"


# Analysis parameters

FPS = 30
PETRI_DIAMETER_MM = 60
BODY_PART = "ABD"
BEE_IDS = ("B1", "B2")


def load_petri_diameter_lookup(summary_file):
    """
    Load Petri dish diameters and create a dyad-level lookup dictionary.

    The Petri dish diameter in pixels is used to calculate a separate
    pixel-to-millimetre conversion factor for each dyad.

    Parameters
    ----------
    summary_file : str
        Path to the CSV file containing the dyad identifiers and measured
        Petri dish diameters in pixels.

    Returns
    -------
    dict
        Dictionary in which each key is a dyad identifier and each value is
        the corresponding Petri dish diameter in pixels.

    Raises
    ------
    FileNotFoundError
        If the summary file does not exist.

    ValueError
        If the required columns are missing.
    """
    if not os.path.exists(summary_file):
        raise FileNotFoundError(
            f"Petri summary file was not found:\n{summary_file}"
        )

    summary_df = pd.read_csv(summary_file)
    summary_df.columns = summary_df.columns.str.strip()

    required_columns = {"Dyad", "Petri_Diameter_px"}
    missing_columns = required_columns.difference(summary_df.columns)

    if missing_columns:
        raise ValueError(
            "The Petri summary file is missing the following columns: "
            f"{sorted(missing_columns)}"
        )

    summary_df["Dyad"] = (
        summary_df["Dyad"]
        .astype(str)
        .str.strip()
    )

    summary_df["Petri_Diameter_px"] = pd.to_numeric(
        summary_df["Petri_Diameter_px"],
        errors="coerce"
    )

    return dict(
        zip(
            summary_df["Dyad"],
            summary_df["Petri_Diameter_px"]
        )
    )


def load_event_lookup(event_file):
    """
    Load crossing and contact timestamps for each dyad.

    Event timestamps are expected to be expressed in seconds from the
    beginning of each video.

    Parameters
    ----------
    event_file : str
        Path to the Excel file containing dyad identifiers and crossing and
        contact timestamps.

    Returns
    -------
    dict
        Nested dictionary containing the crossing and contact timestamps for
        each dyad.

    Raises
    ------
    FileNotFoundError
        If the event file does not exist.

    ValueError
        If the required columns are missing.
    """
    if not os.path.exists(event_file):
        raise FileNotFoundError(
            f"Event file was not found:\n{event_file}"
        )

    events_df = pd.read_excel(event_file)
    events_df.columns = events_df.columns.str.strip()

    required_columns = {"Dyads", "Cross", "Contact"}
    missing_columns = required_columns.difference(events_df.columns)

    if missing_columns:
        raise ValueError(
            "The event file is missing the following columns: "
            f"{sorted(missing_columns)}"
        )

    events_df["Dyads"] = (
        events_df["Dyads"]
        .astype(str)
        .str.strip()
    )

    events_df["Cross"] = pd.to_numeric(
        events_df["Cross"],
        errors="coerce"
    )

    events_df["Contact"] = pd.to_numeric(
        events_df["Contact"],
        errors="coerce"
    )

    event_lookup = {}

    for _, row in events_df.iterrows():
        dyad_id = row["Dyads"]

        event_lookup[dyad_id] = {
            "Cross": row["Cross"],
            "Contact": row["Contact"]
        }

    return event_lookup


def extract_dyad(filename):
    """
    Extract the dyad identifier from a DeepLabCut tracking filename.

    Any text beginning with 'DLC' is removed from the filename. Trailing
    underscores are also removed.

    Example
    -------
    CN17_B1_B2DLC_resnet50_filtered.csv becomes CN17_B1_B2.

    Parameters
    ----------
    filename : str
        Name of the tracking CSV file.

    Returns
    -------
    str
        Extracted dyad identifier.
    """
    base_name = os.path.splitext(filename)[0]
    dyad_id = base_name.split("DLC")[0]

    return dyad_id.rstrip("_")


def find_xy_columns(dataframe, bee_id, body_part="ABD"):
    """
    Find the x and y coordinate columns for one bee and body part.

    The function searches for columns containing a combination such as
    'B1_ABD' or 'B2_ABD'. It supports standard y-column names ending in
    '_y' and previously observed duplicated-column variants containing
    '.1_y'.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Tracking dataframe.

    bee_id : str
        Bee identifier, such as 'B1' or 'B2'.

    body_part : str, default='ABD'
        DeepLabCut body-part label.

    Returns
    -------
    tuple
        The x-column name and y-column name. Returns (None, None) when the
        required columns cannot be found.
    """
    coordinate_label = f"{bee_id}_{body_part}"

    x_columns = [
        column
        for column in dataframe.columns
        if coordinate_label in column
        and column.endswith("_x")
    ]

    y_columns = [
        column
        for column in dataframe.columns
        if coordinate_label in column
        and (
            column.endswith("_y")
            or ".1_y" in column
        )
    ]

    if not x_columns or not y_columns:
        return None, None

    return x_columns[0], y_columns[0]


def calculate_velocity_mm_sec(
    x_coordinates,
    y_coordinates,
    mm_per_pixel,
    fps=30
):
    """
    Calculate instantaneous velocity from consecutive tracking coordinates.

    Euclidean displacement is calculated between each pair of consecutive
    frames. Pixel displacement is converted to millimetres and multiplied by
    the video frame rate to obtain velocity in millimetres per second.

    For a video containing n coordinate frames, the returned velocity array
    contains n - 1 values because each velocity value represents movement
    between two consecutive frames.

    Parameters
    ----------
    x_coordinates : array-like
        Horizontal tracking coordinates in pixels.

    y_coordinates : array-like
        Vertical tracking coordinates in pixels.

    mm_per_pixel : float
        Pixel-to-millimetre conversion factor.

    fps : float, default=30
        Video frame rate in frames per second.

    Returns
    -------
    numpy.ndarray
        Instantaneous velocity values in millimetres per second.
    """
    x_coordinates = pd.to_numeric(
        pd.Series(x_coordinates),
        errors="coerce"
    ).to_numpy()

    y_coordinates = pd.to_numeric(
        pd.Series(y_coordinates),
        errors="coerce"
    ).to_numpy()

    delta_x = np.diff(x_coordinates)
    delta_y = np.diff(y_coordinates)

    displacement_px = np.sqrt(
        delta_x**2 + delta_y**2
    )

    displacement_mm = displacement_px * mm_per_pixel
    velocity_mm_sec = displacement_mm * fps

    return velocity_mm_sec


def safe_nanmean(values):
    """
    Calculate a mean while safely handling empty or entirely missing arrays.

    Parameters
    ----------
    values : array-like
        Values for which the mean should be calculated.

    Returns
    -------
    float
        Mean of the available values. Returns NaN if the input is empty or
        contains no finite observations.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmean(values))


def mean_pre_post_velocity(
    velocity,
    event_time_s,
    fps=30
):
    """
    Calculate mean velocity before and after a behavioural event.

    The event timestamp is converted from seconds to a frame index. Velocity
    values before the event frame are assigned to the pre-event period, while
    values from the event frame onward are assigned to the post-event period.

    Because velocity is calculated between consecutive coordinate frames, its
    length is one value shorter than the original tracking data.

    Parameters
    ----------
    velocity : array-like
        Instantaneous velocity values in millimetres per second.

    event_time_s : float
        Behavioural event timestamp in seconds.

    fps : float, default=30
        Video frame rate in frames per second.

    Returns
    -------
    tuple
        A tuple containing:

        - Mean pre-event velocity
        - Mean post-event velocity
        - Event frame index
    """
    velocity = np.asarray(velocity, dtype=float)

    if pd.isna(event_time_s):
        return np.nan, np.nan, np.nan

    event_frame = int(
        round(float(event_time_s) * fps)
    )

    if event_frame <= 0:
        pre_event_mean = np.nan
        post_event_mean = safe_nanmean(velocity)

    elif event_frame >= len(velocity):
        pre_event_mean = safe_nanmean(velocity)
        post_event_mean = np.nan

        print(
            f"Warning: event frame {event_frame} is outside the available "
            f"velocity range of {len(velocity)} values."
        )

    else:
        pre_event_mean = safe_nanmean(
            velocity[:event_frame]
        )

        post_event_mean = safe_nanmean(
            velocity[event_frame:]
        )

    return (
        pre_event_mean,
        post_event_mean,
        event_frame
    )


def process_tracking_file(
    file_path,
    dyad_id,
    petri_diameter_px,
    event_information
):
    """
    Process one dyad tracking file.

    Velocity is calculated separately for B1 and B2 using abdomen coordinates.
    Mean velocities are then calculated before and after crossing and contact.

    Parameters
    ----------
    file_path : str
        Path to the filtered DeepLabCut tracking CSV file.

    dyad_id : str
        Dyad identifier.

    petri_diameter_px : float
        Petri dish diameter in pixels for the current dyad.

    event_information : dict
        Dictionary containing crossing and contact timestamps.

    Returns
    -------
    list of dict
        Event-phase velocity results for the bees successfully identified in
        the tracking file.
    """
    dataframe = pd.read_csv(
        file_path,
        low_memory=False
    )

    dataframe.columns = dataframe.columns.str.strip()

    mm_per_pixel = (
        PETRI_DIAMETER_MM / petri_diameter_px
    )

    cross_time_s = event_information["Cross"]
    contact_time_s = event_information["Contact"]

    file_results = []

    for bee_id in BEE_IDS:
        x_column, y_column = find_xy_columns(
            dataframe=dataframe,
            bee_id=bee_id,
            body_part=BODY_PART
        )

        if x_column is None or y_column is None:
            print(
                f"Skipping {os.path.basename(file_path)}, {bee_id}: "
                f"{BODY_PART} coordinate columns were not found."
            )
            continue

        velocity = calculate_velocity_mm_sec(
            x_coordinates=dataframe[x_column],
            y_coordinates=dataframe[y_column],
            mm_per_pixel=mm_per_pixel,
            fps=FPS
        )

        cross_pre, cross_post, cross_frame = (
            mean_pre_post_velocity(
                velocity=velocity,
                event_time_s=cross_time_s,
                fps=FPS
            )
        )

        contact_pre, contact_post, contact_frame = (
            mean_pre_post_velocity(
                velocity=velocity,
                event_time_s=contact_time_s,
                fps=FPS
            )
        )

        file_results.append(
            {
                "Dyad": dyad_id,
                "Bee_ID": bee_id,
                "Cross_Time_s": cross_time_s,
                "Contact_Time_s": contact_time_s,
                "Cross_Pre_mm_s": (
                    round(cross_pre, 2)
                    if pd.notna(cross_pre)
                    else np.nan
                ),
                "Cross_Post_mm_s": (
                    round(cross_post, 2)
                    if pd.notna(cross_post)
                    else np.nan
                ),
                "Contact_Pre_mm_s": (
                    round(contact_pre, 2)
                    if pd.notna(contact_pre)
                    else np.nan
                ),
                "Contact_Post_mm_s": (
                    round(contact_post, 2)
                    if pd.notna(contact_post)
                    else np.nan
                ),
                "Cross_Frame": cross_frame,
                "Contact_Frame": contact_frame
            }
        )

    return file_results


def main():
    """
    Run the complete event-phase velocity analysis.
    """
    if not os.path.isdir(TRACKING_FOLDER):
        raise FileNotFoundError(
            "The tracking-data folder was not found:\n"
            f"{TRACKING_FOLDER}"
        )

    output_folder = os.path.dirname(
        OUTPUT_EXCEL
    )

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    petri_lookup = load_petri_diameter_lookup(
        PETRI_SUMMARY_FILE
    )

    event_lookup = load_event_lookup(
        EVENT_FILE
    )

    results = []

    tracking_files = sorted(
        filename
        for filename in os.listdir(TRACKING_FOLDER)
        if filename.lower().endswith(".csv")
    )

    if not tracking_files:
        raise FileNotFoundError(
            "No CSV tracking files were found in:\n"
            f"{TRACKING_FOLDER}"
        )

    print(
        f"Found {len(tracking_files)} CSV files."
    )

    for filename in tracking_files:
        file_path = os.path.join(
            TRACKING_FOLDER,
            filename
        )

        dyad_id = extract_dyad(
            filename
        )

        if dyad_id not in petri_lookup:
            print(
                f"Skipping {filename}: dyad '{dyad_id}' was not found "
                "in the Petri summary."
            )
            continue

        if dyad_id not in event_lookup:
            print(
                f"Skipping {filename}: dyad '{dyad_id}' was not found "
                "in the crossing/contact file."
            )
            continue

        petri_diameter_px = petri_lookup[
            dyad_id
        ]

        if (
            pd.isna(petri_diameter_px)
            or petri_diameter_px <= 0
        ):
            print(
                f"Skipping {filename}: invalid Petri dish diameter "
                f"({petri_diameter_px})."
            )
            continue

        print(
            f"Processing {dyad_id}..."
        )

        file_results = process_tracking_file(
            file_path=file_path,
            dyad_id=dyad_id,
            petri_diameter_px=petri_diameter_px,
            event_information=event_lookup[dyad_id]
        )

        results.extend(
            file_results
        )

    if not results:
        raise RuntimeError(
            "The analysis produced no results. Check the dyad names, "
            "tracking-column names, event file, and Petri summary file."
        )

    results_df = pd.DataFrame(
        results
    )

    output_columns = [
        "Dyad",
        "Bee_ID",
        "Cross_Time_s",
        "Contact_Time_s",
        "Cross_Pre_mm_s",
        "Cross_Post_mm_s",
        "Contact_Pre_mm_s",
        "Contact_Post_mm_s",
        "Cross_Frame",
        "Contact_Frame"
    ]

    results_df = results_df[
        output_columns
    ]

    results_df = results_df.sort_values(
        by=["Dyad", "Bee_ID"]
    ).reset_index(
        drop=True
    )

    results_df.to_excel(
        OUTPUT_EXCEL,
        index=False
    )

    print()
    print("Analysis completed successfully.")
    print(f"Output file: {OUTPUT_EXCEL}")
    print(f"Rows exported: {len(results_df)}")
    print(
        f"Dyads represented: "
        f"{results_df['Dyad'].nunique()}"
    )


if __name__ == "__main__":
    main()
