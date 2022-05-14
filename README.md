# Learning-Model-Predictive-Control-Using-Dynamic-Bicycle-Model-with-RBF-Based-Gaussian-Process

How to implement:
1. First keep the parameters as unchanged in the ftocp file and run the samplecollect file to collect data for the parameters-unchaged model.
2. Then, change the parameter you want to change in the ftocp file and run the samplecollect file to collect data for the parameters-chaged model.
3. Using above 2 sets of collected data, use the gp-pretrain file to train the data 3 times.(each time set the VARIDX to be 3,4,5 and we need to train 3 times because they are for states of vx, vy and omega respectively.)
4. Change the same parameter in the gp_ftocp file as the parameter changed in the ftocp file.
5. Using the gp_mpc to implement the GP-corrected data collected from step 4 to the dynamic bicycle model. 

Reminder: keep the parameters as unchanged in the utils file. 
