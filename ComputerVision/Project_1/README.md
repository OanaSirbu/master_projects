# Computer Vision - Project 1 / Mathable Score Calculator
### Sirbu Oana-Adriana, 407 AI

## Installation

To run Mathable score calculator, you will need the following library versions:

- python==3.10.9
- opencv-python==4.9.0.80
- numpy==1.26.4

## How to Run the Solution

To run the Mathable score calculator on your computer, follow these steps:

1. Open the Python file `mathable_game_scorer_Sirbu_Oana.py`.
2. Modify the following paths in the file to match your environment:
    - Line 11: Change the path to indicate your working directory (the one which has the folder with images).
    - Lines 14, 15, 16: Change the path to the indicated files (all can be found in the training data provided; they are also attached to the submission).
    - Line 21: Modify the path to reach the folder with the test images and corresponding annotated .txt files.
    - Lines 24, 25, 26 & 28: Define paths in which the folders containing the preprocessed data will be stored.
    - Line 31: Indicates the output folder, where the predictions of the algorithm will be stored (all the folders associated with these paths will be automatically created by the code, they just need to be specified before).
3. If the numbers of the games are different than 1, 2, 3, and 4, adjust the following lines of code:
    - Line 39: Update the list accordingly.
    - Lines 560, 825: Update the range accordingly.
4. Once all necessary changes are made, save the file.
5. Run the Python file. It will automatically perform all the operations.
