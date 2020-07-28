# ai_unknown_transitions

The files in this repository reproduce the simulations in the master thesis "Discrete state-space active inference and unknown controlled state transitions" by Luuk van Roessel, which can be found in the repository of the Delft University of Technology.

The simulation files were based on standard routines of Statistical Parametric Mapping (SPM) which is distributed under the terms of the GNU General Public Licence as published by the Free Software Foundation, hence this work redistributed under the same terms.

To run the simulations:
1. Install SPM12 (https://www.fil.ion.ucl.ac.uk/spm/)
2. Start SPM12
3. Place the files of this repository in the following SPM directory "..\SPM\spm12\toolbox\DEM"
4. Open "DEMO_MDP_maze_unknown_transitions", choose a scenario and run 

Most important modifications are:

SPM_MDP_VB_X_unknown_transitions, line 653-667: Implementation of the transition novelty term                                 --> Thesis section 3-2

SPM_MDP_VB_X_unknown_transitions, line 806-813: Implementation of the alternative learning mechanism                          --> Thesis section 5-3

DEMO_MDP_MDP_maze_unknown_transitions, line 175-188: Implementation of the transition uncertainty                             --> Thesis section 4-3

DEMO_MDP_MDP_maze_unknown_transitions, line 368-373: Implementation of the incorrect concentration parameter removal mechanism--> Thesis page 52
