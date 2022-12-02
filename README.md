# UdacityNavigation

## Getting Started

Follow the instructions below to set up your python environment to run the code in this repository.
Note that the installation was only tested for __Mac__.

1. Initialize the git submodules in the root folder of this repository. 

	```bash
	git submodule init
	git submodule update
	```
 
2. Create (and activate) a new conda environment with Python 3.6.

	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	
3. Install the base Gym library and the **box2d** environment group:

	```bash
	pip install gym
	pip install Box2D gym
	```

4. Navigate to the `external/Value-based-methods/python/` folder.  Then, install several dependencies.

    ```bash
    cd external/Value-based-methods/python
    pip install .
    ```
    **Note**: If an error appears during installation regarding the torch version, please remove the torch version 0.4.0 in
    external/Value-based-methods/python/requirements.txt and try again.

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    **Note**: Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

6. Download the [Unity environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)  in the **root folder** of this repository.

    