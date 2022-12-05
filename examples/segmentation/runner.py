"""
File: runner.py
------------------
Runner script to train the model. This is the script that calls the other modules.
Execute this one to execute the program! 
"""


import configs
import argparse
import warnings
import pdb 
import experiments


def main(args):
    if args.config is None:
        config_class = 'BaseConfig'
    else:
        config_class = args.config
    cfg = getattr(configs, config_class)
    exp = cfg.experiment(
        config=cfg
    )

	# train the model
    exp.test() 
    exp.train(num_epochs=15)
    exp.test()



if __name__ == '__main__':
    # configure args 
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-config", type=str, help='specify config.py class to use.') 
    args = parser.parse_args()
    main(args)
	