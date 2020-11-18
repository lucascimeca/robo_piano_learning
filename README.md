# Robot Piano Learning - 25/05/2020

### Authors: Luca Scimeca (luca.scimeca@live.com) & Cheryn Ng (cheryn97@gmail.com)

![Alt Text](https://github.com/lucascimeca/robo_piano_learning/blob/master/assets/piano_playing_palpation_short.gif)


This repository contains code and data for the "Robot Piano Learning" project in the Engineering Department of the University of Cambridge, under the supervision of Fumiya Iida. 

all code in the project is contained in the "src" folder. The data is contained in the "data" folder. Each sub-folder within the data folder contains a different set of experiments. A simple demontration of the use of the data within the data folder is provided in the "src/scripts" subfolder (see below).

### Dependencies (libraries)
-  python 3.7
 - numpy 
 - GPFlow 2
 - sklearn
 - matplotlib
 - pandas 
 - json
 - mido

##### -- For a simple example of how to load and use the robot control and corresponding midi outputs see the code in "src/scripts/simple_load.py"
##### -- For a virtual run of the experiments run the scripts at "src/scripts/virtual_run.py"
##### -- To re-generate the figures in the publication, virtual-run the experiments with the appropriate parameters and run the "src/scripts/generate_figures.py" script file on the generated results
