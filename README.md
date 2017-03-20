This package contains an EMC and a Difference map implementation for simS2E/SIMEX.

The EMC module
==============
Contains
--------
    start_orient
    compile_EMC
    EMC.c
    runEMC.py
    make_diagnostic_figures.py
    [dir]quaternions
    [dir]supp_py_modules

Requirements
------------
    gcc, cmath library
    Python2.7
    Cython
    openmp

Usage
-----
Use start_orient to automatically start the EMC module in simS2E.
Here's a summary of what start_orient does
./start_orient
    - uses "compile_EMC" to compile EMC.c into executable "EMC"
    - uses "compile_EMC" to compile Cython files in supp_py_modules
    - starts orientation recovery with Python script "runEMC.py"
        runEMC.py
            - converts h5 diffraction data to sparse photon.dat in tmp dir.
            - also creates detector.dat in tmp dir.
            - uses EMC executable iteratively and saves output in tmp dir.
            - returns simS2E standard output in orient_YYYY_MM_DD_hh_mm_ss.h5

Quick start
-----------
Modify "start_orient" to customize orientation recovery. Here are some options, in brief

The file directories should almost certainly be modified:
	SRC: 	Location of source code for reconstruction (e.g. EMC.c runEMC.py)
	INPUT:	Location of input data (diffraction images upstream along simS2E)
	TMP:	Where intermediate EMC output should be stored
	OUTPUT:	Where output should be stored for the next step in simS2E

Other options include:
	INITIALQUAT: 	starting level for quaternion refinement
	MAXQUAT:		final level of quaternion refinement
	MAXITER:		maximum number of EMC iterations for intensity reconstruction
	MINERR:			(in quotes) the change in 3D model between iterations below which either
					a finer rotation group sampling is chosen or EMC iteration
					terminates if no finer sampling is available
	BEAMSTOP:		0 for no beamstop, 1 for circular plus strip beamstop
	PLOT:			model error and convergence plots generated as EMC iterates
	DETAILED:		(caution: consumes much disk storage)
					outputs intermediate 3D volumes of the reconstruction
	SLEEPDUR:		number of seconds to pause between reconstructions
	NUMRECON:		number of independent EMC reconstructions to attempt

For other options, compare the fields in start_orient with the usage help from:
	python runEMC.py -h


The Difference map phase retrieval module
=========================================

Contains
--------
    start_phasing
    compile_DM
    object_recon.c
    runDM.py

Requirements
------------
    gcc, cmath library
    fftw
    Python2.7


Usage
-----
Use start_phasing to automatically start the phasing module in simS2E.
Here's a summary of what start_phasing does
./start_phasing
    - uses "compile_DM" to compile object_recon.c into executable
      "object_recon"
    - uses "runDM.py" to start iterative phase retrieval.
        runDM.py
            - computes a crude initial support based on 2-means clustering of the
			autocorrelation function
			- saves crude initial support to support.dat
			- starts shrink_wrap + iterative phase retrieval using this support and other parameters (listed below)
			- saves intermediate output to file

Quick start
-----------
Modify "start_phasing" to customize iterative phase retrieval. Here are some options, in brief

The file directories should almost certainly be modified:
	SRC: 	Location of source code for phasing (e.g. object_recon.c runDM.py)
	INPUT:	Location of input 3D diffraction volume
	TMP:	Where intermediate phasing output should be stored
	OUTPUT:	Where output should be stored simS2E

Other options include:
	NUMTRIALS:		number of random restarts for each support size
	STARTAVE:		support averaging starts after these many iterations for
					each randomly restarted reconstruction trial
	LEASH:			leash parameter for the modified difference map
	SHRINKCYCLES:	number of times to run the shrink_wrap support-shrinking algorithm
	SLEEPDUR:		number of seconds to pause between reconstructions
	NUMRECON:		number of independent phasing reconstructions to attempt

For other options, compare the fields in start_phasing with the usage help from:
	python runDM.py -h
