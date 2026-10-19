# Synthetic intent starter data

`starter.csv` is a small, manually authored **synthetic** dataset used only to make the future
intent pipeline executable. It is not representative of real hotel traffic and must not be used
to claim real-world model performance.

`collected.csv` adds 72 hand-authored synthetic English, German, and French examples. Its fixed
`split` column has 36 training, 18 development and 18 test rows (three per intent in the test).
The 30 starter examples belong to training. The trainer rejects duplicate text and selects
features on development examples before reporting test results. These messages were authored
for this project; none came from guests or a licensed public corpus. Keep future guest data out
of the repository and obtain consent and a retention policy before using it for training.

