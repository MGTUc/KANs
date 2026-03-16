The *requirement.txt* file has the versions I used in the development but I believe the code should work with more recent updates.

Everything should run from *ss_kan_main.py* with *model_state_space.py* having the class that creates the torch state-space model. 

The *_plot.py* sciprt has a lot of plotting functions that I used in the code developement and most of them are test case dependent. The ones that are a bit more general are called in the postprocessing script *evaluate_ss_kan.py* which assumes a trained model. 
