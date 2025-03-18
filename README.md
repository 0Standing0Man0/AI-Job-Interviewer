# AI-Job-Interviewer
This project is a simple implementation of a job interviewer system using Python. The system will ask the user a total of 5 HR questions by default and display their score based on Posture and Communication Skills.

## Model
To detect posture, we utilized a VGGNet-16 model, fine-tuned on a private dataset. This resulted in an F1-score of 1.0.

[Trained VGGNet-16 Model](https://drive.google.com/file/d/1Xiyyr1PHBvNaPnPl_b-4ljQ60y2b6DDy/view?usp=sharing)

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/0Standing0Man0/AI-Job-Interviewer.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd AI-Job-Interviewer
    ```

3.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  Download the provided VGGNet-16 model and place it in the project's root directory

## Usage

1.  Open `Code/hugging_face_api.py` and replace `"YOUR_API_HERE"` with your Hugging Face access token

2.  Run the application:

    ```bash
    python Code/app.py
    ```

3. Find "Running on http://127.0.0.1:PORT_NUMBER" in the terminal and open it in your web browser

4. Follow the instructions on the screen to answer the HR questions. The system will analyze your posture and communication skills and display your score