import time


def show_progress(
    current_row_i: int, total_rows: int, imu_time: float, start_time: int
) -> None:
    """
    Outputs a status into the terminal showing the progress through the loop
    in `main` in `render.py`

    e.g. "20/6959 (0.2874%) | 0.070312 | 3.8"
    """

    # The index of the current row of data in the dataset
    current_row_i += 1

    fraction = f"{current_row_i+1}/{total_rows}"
    percentage = round(((current_row_i + 1) / total_rows) * 100, 4)

    # The time elapsed since the iterating began
    time_elapsed = round(time.time() - start_time, 2)

    print(f"{fraction} ({percentage}%) | {imu_time} | {time_elapsed}")
