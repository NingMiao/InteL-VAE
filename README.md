# InteL-VAEs
Codes for paper &lt;On Incorporating Inductive Biases into VAEs>. InteL-VAE is a simple and effective method for learning VAEs with controllable inductive biases by using an intermediary set of latent variables. It allows us to impose desired properties like sparsity or clustering on learned representations, and incorporate prior information into the learned model.
![Model Graph](https://github.com/djkdsjwkjerkjermf/InteL-VAE/blob/master/model.png)

## Usage
### To try low dimensional datasets, 
	run *_low_dim.ipynb in Jupyter notebook.
### To train inteL-VAEs and other baselines,
	run train_*.sh 
Hyper-parameters can be changed in .sh files.
### To run downstream tasks,
	run downstream_*.sh
Please run downstream tasks after training corresponding VAEs.
 
## Requirements
 - Tensorflow `>= 2.2.0`
 - sklearn (Only for downstream tasks.)
 - Pillow (PIL)
 - fid_score (Only for calculating FID scores.)
