# Physics-based data-driven modeling
![alt tag](docs/schematics.png)
## Objective
The objective for this work is to develop a data-driven proxy to high-fidelity numerical flow simulations using digital images. 
The proposed model can capture the flow field and permeability in a large verity of digital porous media based on solid grain geometry and pore size distribution by detailed analyses of the local pore geometry and the local flow fields. 
To develop the model, the detailed pore space geometry and simulation runs data from 3500 two-dimensional high-fidelity Lattice Boltzmann simulation runs are used to train and to predict the solutions with a high accuracy in much less computational time. 
The proposed methodology harness the enormous amount of generated data from high-fidelity flow simulations to decode the often under-utilized patterns in simulations and to accurately predict solutions to new cases.
The developed model can truly capture the physics of the problem and enhance prediction capabilities of the simulations at a much lower cost. 

## Architecture set-up
The proposed network consists of 6 strided convolutions which reduce the size of input by a factor of 64, followed by four residual blocks.  Each  residual  block  consists of  two  3x3  convolutional  layers. Finally, In  order  to  obtain  the  final  prediction map, we add six subsequent up-sampling blocks on top of the residual blocks. The input to each up-sampling block is the feature maps of the previous layer concatenated in depth with those of contracting down-sampling path. As mentioned earlier each up-sampling block comprises successive NN-upsampling and 3x3 convolution with unit stride. Note that all the convolutions are followed by a Batch Normalization and Relu activation function. 
 Since our input (geometry input image) is the simulated velocity fields we employ reflection 1x1 padding for all the convolutions.
![alt tag](docs/net.png)



Side-by-side comparison of the model predictions vs. LB simulations results (ground truth) for four test cases. (a) input images used for simulations; (b) model predictions; (c) LB simulation results (ground truth); (d) error percentage between (b) and (c); and (e) distribution of error in each case.
![alt tag](docs/error.png)
