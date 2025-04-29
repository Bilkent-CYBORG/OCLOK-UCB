

# Contextual Combinatorial Bandits With Changing Action Sets Via Gaussian Processes

This repository is the official implementation of Contextual Combinatorial Bandits With Changing Action Sets Via Gaussian Processes, submitted to Machine Learning.
![Illustration of our algorithm called O'CLOCK-UCB.](https://am3pap005files.storage.live.com/y4mhr7YuNEW5H7WHwwDKXon9asOz6h7FH3ptlUg_DNAxmXnw9SA84fGhlbwtkGpTjOpbVFCJl8PpHE5JfA7kUs4MyFZtf7Fwe2JavetySkUID6DyXij1vW6xOEdsPN6AfKlsCugPNwiRWdJRNIPZX5UBgskUwRFzxU6vXF4Ktnpn9E1g3iUGMVOOyXm45IjYwo3fdHYQcBkVclnow0hLlXueA/algo_illustration.png?psid=1&width=1483&height=639)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

We use the [gpflow](https://github.com/GPflow/GPflow) library for all GP-related computations and gpflow uses tensorflow. Our code uses the TIM+ algorithm, for which you must link the C++ TIM+ code to Python. Follow [here](https://github.com/altugkarakurt/OCIMP) for linking instructions. Once the library has been generated, place it both in the root directory where main.py is and also inside the tim_plus directory.

## Running the simulations
We ran a total of three simulations. Moreover, none of the algorithms that we implement and test do offline-learning, thus there is no 'training' to be done. However, to be able to repeat the simulations and also improve speed, we first generate the arm contexts, rewards, and other setup-related information and save them as HDF5, in the case of Simulation I, and pickled DataFrames, in the case of Simulations II & III. We provide the links to the generated datasets that we used at the bottom of this README file. By default, when you run the script (main.py), it re-generates new datasets and runs the simulations on them.

### Simulation I (movie recommendation)
To run Simulation I, provide the argument `sim_1` to the main.py script. For example, to re-generate this simulation's datasets and run the simulations on the newly generated datasets use

```
python main.py main
```
and to run the main paper simulations using pre-generated datasets , which must be in the root directory, use
```
python main.py main --use_saved_dataset
```
### Simulation II (Foursquare)
To run Simulation II, use the argument `sim_2`. You can provide the `--use_saved_dataset` argument to use pre-generated and saved datasets.
### Simulation III (Varying kernel parameters)
To run Simulation III, use the argument `sim_3`. You can provide the `--use_saved_dataset` argument to use pre-generated and saved datasets.

## Generating plots and figures

After the script has run the simulations, it will automatically plot the reward and regret curves and save them as PDFs. Then, if you want to re-generate the plots without running the whole simulation again, you can give the `--only_plot` argument to the main.py script. 

## Generated datasets

You can download the generated datasets that we ran the simulations with below:

### Simulation I dataset
The HDF5 dataset file can be downloaded [here](https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdjctellURkxybzBycnhlcmVZeGQtWXpHZW85VFE_ZT1uZjlxQnk/root/content). Make sure to place the 'movielens_simulation.hdf5' file in the root directory where main.py is.

### Simulation II datasets
A zip file of the pickled DataFrames used for the Foursquare simulation (Simulation II) can be downloaded [here](https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdjctellURkxybzBycnhjUmJBamFwLV8wQ010aVE_ZT1sWmJWODA/root/content). Make sure to extract both 'fs_tky_simulation_df_uni' and 'fs_tky_simulation_df_nuni' and place them in the root directory where main.py is.

We use the Wolfram Engine to learn the distribution of the TKY dataset's locations; thus to generate the dataset, you must have the Wolfram Engine installed. It can be installed for free [here](https://www.wolfram.com/engine/). Moreover, you must download the exported LearnedDistribution file available [here](https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdjctellURkxybzBycnhoTFFxWEY5dXJybkNRWVE_ZT1JWVRoelE/root/content) and set its absolute path in fs_problem_model.py. Note that if you download the saved datasets (‘fs_tky_simulation_df_uni’ and ‘fs_tky_simulation_df_nuni’), you will NOT need to download the Wolfram Engine. The Wolfram Engine is only needed to generate the datasets.

### Simulation III datasets
A zip file of the pickled DataFrames used for the varying arm codependence simulation (Simulation III) can be downloaded [here](https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdjctellURkxybzBycnhhcVgtY2dDYlRfb1BOdHc_ZT1OUDA3Nlg/root/content). Make sure to extract all of the five files, each corresponding to a different kernel lengthscale, and place them in the root directory where main.py is.

## Results
Our algorithm beats the s.o.t.a contextual combinatorial multi-armed bandits with changing action sets (C3-MAB) algorithm, [ACC-UCB](http://proceedings.mlr.press/v108/nika20a.html). The figure below shows the time-averaged reward of the sparse version of our algorithm (SO'CLOK_UCB) and ACC-UCB on a movie-recommendation simulation (Simulation I) using the [MovieLens dataset](https://grouplens.org/datasets/movielens/). Notice that even with just 2 inducing points, we manage to outperform ACC-UCB. See Section 5 of the paper for a detailed explanation of the setup and in-depth analysis.
![Main results figure](https://am3pap005files.storage.live.com/y4mLDE9PD4XurcEUSR9hph9xx2m7j5Ch72nKYimhRNvay-lqnXbvUOQQMMMZtnxeaYX851sOIzyhRyVUCqf9wDruGSk_NVEb8ZbwDrKCdrRBe3Xk2HjtoNPtMrsuTlXD_y8kOozXr4HRXGU9L33OMyTt1gUgOdy5sPmBJtlO_y_jIPlA32dZT-cqVgkNzmBARqQ?width=1192&height=715&cropmode=none)
