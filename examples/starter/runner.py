"""
File: runner.py
------------------
Driver script which runs the program (i.e., 
runs the experiment). 
"""

import experiments 


def main():
	exp = experiments.BaseExp()
	exp.train(num_epochs=10)
	exp.test()
	print("Experiment complete!")


if __name__ == '__main__':
    main()
