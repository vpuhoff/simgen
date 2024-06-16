### Facial Beauty Predictor

This application captures a specified screen region, analyzes the facial beauty using a pre-trained ComboNet model on the SCUT-FBP5500 dataset, and displays the results on the screen. Additionally, it can simulate mouse clicks and play a sound based on the analysis results.

#### Features

- Captures a specified screen region.
- Analyzes facial beauty using a pre-trained ComboNet model on the SCUT-FBP5500 dataset.
- Displays the analysis results in the top-right corner of the screen.
- Simulates mouse clicks if the beauty score is below a threshold.
- Plays a sound if the beauty score is above a threshold.
- Controlled via hotkeys to start and stop the capture and analysis process.

#### Requirements

- Python 3.x
- `torch`
- `torchvision`
- `Pillow`
- `pyautogui`
- `keyboard`
- `pygame`
- `tkinter`
- `numpy`

#### Installation

1. Clone the repository or download the source code.
2. Install the required Python packages:
    ```sh
    pip install torch torchvision pillow pyautogui keyboard pygame numpy
    ```
    Note: `tkinter` is included with Python, but you may need to install it separately on some systems.

#### Pre-trained Model

The application uses the ComboNet model pre-trained on the SCUT-FBP5500 dataset. You can download the pre-trained model from the [SCUT-FBP5500 dataset repository](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). Ensure you have the model file named `ComboNet_SCUTFBP5500.pth` in the same directory as the script.

#### Usage

1. Ensure you have the pre-trained model file `ComboNet_SCUTFBP5500.pth` in the same directory as the script.
2. Replace `'alert_sound.mp3'` with the path to your desired sound file for the alert.
3. Run the script:
    ```sh
    python facial_beauty_predictor.py
    ```

#### Hotkeys

- `ALT+F1`: Start capturing and analyzing the screen region.
- `ALT+F2`: Stop capturing and analyzing.
- `ESC`: Exit the application.

#### Configuration

- The screen region to capture is defined by the `region` variable in the `main` function:
    ```python
    region = (850, 50, 850, 850)  # (x, y, width, height)
    ```
- The coordinates for the mouse click are defined in the `start_capture` function:
    ```python
    pyautogui.click(x=1200, y=1054)
    ```
- The threshold for the beauty score to simulate a mouse click is set to 3.8:
    ```python
    if result['beauty'] < 3.8:
    ```
- The sound file for the alert is specified in the `start_capture` function:
    ```python
    pygame.mixer.music.load('alert_sound.mp3')
    ```

#### Code Overview

The main components of the script are:

1. **FacialBeautyPredictor Class**: Initializes the model, loads the pre-trained weights, and provides a method to infer the beauty score from an image.
2. **capture_screen Function**: Captures a screenshot of the specified region.
3. **start_capture Function**: Continuously captures the screen, analyzes the beauty score, updates the GUI, and performs actions based on the score.
4. **main Function**: Sets up the hotkeys, initializes the GUI, and starts the main loop.

### Example

Here is an example of the output:

```
Press ALT+F1 to start capturing and analyzing. Press ALT+F2 to stop.
Beauty Score: 3.45, Time Elapsed: 1.23 seconds
Beauty Score: 4.02, Time Elapsed: 1.21 seconds
```

The GUI will display the current beauty score in the top-right corner of the screen.

---

This README provides an overview of the application, including installation instructions, usage guidelines, and configuration options. If you have any questions or issues, please refer to the code comments or contact the repository maintainer.