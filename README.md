# Assignment 2 – AI & ML for Robotics (PyBullet Project)

## Features
- Markov Localization (particle filter, EKF, PL-ICP correction)
- Grid-based SLAM (occupancy mapping)
- D* Path Planning (dynamic replan, grid search)
- RL Agent for exploration strategy
- Fully modular, extendable, and relates directly to supplied research paper

## Running
1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Run simulation:
    ```
    python main.py
    ```
- Press Ctrl+C to stop.
- Take a FULL SCREENSHOT showing GUI, code output, and system date/time for submission.

## RL Justification
RL is used to optimize exploration, reducing map uncertainty, similar to maximizing information gain as referenced in the paper.

## Screenshots
Required for submission, as per instructions.

## Extending
You can easily plug in more advanced sensor fusion, mapping, or planning logic, or richer action/state spaces.

---

