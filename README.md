# Learned Experience Replay
The code for the final project for CS221. We implemented a learned replay buffer, as well as a baseline agent with a normal replay buffer, and an oracle agent with prioritized experience replay.

To run, assuming all the packages are installed using 'pip install -r requirements.txt', all one needs to do is run 'python main.py'

To use prioritized experience replay, in the 'd3qn.py' file, change 'self.weighted' to True on line 61 in the Agent class.

To use prioritized experience replay, in the 'd3qn.py' file, change 'self.learned' to True on line 61 in the Agent class. Ensure 'self.weighted' is set to False.

The names for the files at the bottom of 'main.py' can also be changed. This will save the plot using the name entered.

Only tested on python3.6+