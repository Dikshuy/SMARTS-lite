put this directory (abil) into the SMARTS directory;

to train, put following command in supervisord.conf and run supervisord:
command=python abil/rllib_single_train.py --headless scenarios/loop

You can change training parameters, like training time or batch size, in the code itself. 
When training ends, model is saved into <parent directory of rllib_single_train.py>/model

to load the saved model and test it, first make sure that "path_to_model" in rllib_agent.py specifies the correct path
to the saved model, then put following command into supervisord.conf and run supervisord:
command=python abil/rllib_single_test.py scenarios/loop

