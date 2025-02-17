## Cpyquaticus ##
This code is an implementation of a  basic MCTF game in c taken from Pyquaticus Simulation Library. 
For all the nice features and custom relative observation space use: https://github.com/mit-ll-trusted-autonomy/pyquaticus 

This Library allows high amounts of environment sampling (On one macbook pro ARM2 core you can achieve 10k steps per second)

### Limitations ###
Currently, this only allows for the normalized or unnormalized actual positions in the observations space, which, which currently does not include relative observation support (TODO).

## Installation ##

Clone the repository
cd into Cpyquaticus
pip install -e .

## Training ##
Clone RL Training Library:
git clone https://grimyRL
Follow installation steps on the grimyRL library page
cd into the Cpyquaticus rl_test folder
Run: 
python petting_zoo_ppo_train.py


