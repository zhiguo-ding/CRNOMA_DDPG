# CRNOMA_DDPG
These are the codes for the following paper

Z. Ding, R. Schober, and H. V. Poor, No-Pain No-Gain: DRL Assisted Optimization in Energy-Constrained CR-NOMA Networks, IEEE Trans. Communications, submitted. (a copy of this paper is included in this folder)


A good starting point is to run the file 'two_user_case.py', which is fast and generates Fig. 1 in the paper. In order to simulate the case with fading, the file 'special_case.py' should be used, where Figs. 2, 3, and 4 in the paper can be generated. The file 'random_location_with_fading K.py' will take quite long time to run.   

---------------------------------------
The paper and the codes have been updated to ensure that the DDPG algorithm can also work in the case with time-varying channels. The revision of the paper and the codes can be found in the timevarying folder. A good starting point is to run the file 'special_case.py', which generates Fig. 5 in the revised paper. The use of file, 'fading K varying.py', generates Fig. 6, but it takes quite long time to run. 
