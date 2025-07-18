## [Preference Discerning with LLM-Enhanced Generative Retrieval](https://arxiv.org/abs/2412.08604)

This repository contains code for the paper [Preference Discerning with LLM-Enhanced Generative Retrieval](https://arxiv.org/abs/2412.08604), including dataset generation for our evaluation benchmark, training RQ-VAE for semantic id generation, and training of multi-modal generative retrieval models.
First, clone the repository and create a conde environment via

    git clone https://github.com/facebookresearch/prefdisc.git
    cd prefdisc
    conda env create -f env.yml

Alternatively, you can install all package dependencies via

    pip install requirements.txt

The structure of this repository is as follows

    .
    ├── configs                    # Contains all .yaml config files for Hydra to configure dataloading, train sequence generation, models, etc.
    │   ├── setting
    │   ├── logging
    │   └── main.yaml              # Main config file for Hydra
    ├── dataset                    # Contains scripts for running our preference generation and postprocessing pipelines
    ├── GenRet                     # contains src for training generative retrieval models
    ├── ID_generation              # Scrips for running experiments on Slurm/PBS in multi-gpu/node setups.
        ├── models                 # contains RQ-VAE model for semantic id generation
        ├── preproecessing         # contains scripts for data preprocessing
            ├── processed          # directory where processed user sequences are stored
    ├── results                    # contains pickled results reported in the paper
    ├── LICENSE
    ├── README.md
    ├── environment.yaml
    ├── requirements.txt
    ├── utils.py                   # utils for training
    └── run.py                     # Main entry point for training


The configs directory contains all configurations for running and reproducing all the experiments of our paper.

### Preference Approximation

To reproduce our preference generation pipeline, navigate to the ```dataset``` directory and execute

    python approximate_preferences.py

This will automatically load the model weights of LLama3-70B-Instruct and use our pre-defined prompt templates to generate the user preferences for all the different datasets we used in the paper.
Please be aware that this script is not optimized for speed and may take a while unless you are able to parallelize across multiple model instances as we did.
Also consider using optimized libraries for that process, such as [vLLM](https://github.com/vllm-project/vllm).
The generation parameters in the script are being set to the default ones we used in our work.
After generation, the script will automatically run our postprocessing pipeline and dump the generated user preferences as a json file.

The first step after compiling the user preferences is to match them to ground truth items for the sequential recommendation task via

    python match_preferences_to_items.py --dataset Beauty

The ```--dataset``` command line argument can take any of the five datasets used in this work.
By default this script uses a pre-trained Sentence-T5-XXL for the matching process.
Next, you will need to perform sentiment classification on the item reviews and the generated user preferences via

    python get_pos_neg_preferences.py --dataset Beauty
    python get_pos_neg_review.py --dataset Beauty

Subsequently, you can generate the evaluation benchmarks for sentiment following, preference steering and preference consolidation via

    python generate_pos_neg_split.py --dataset Beauty
    python generate_fine_coarse_split.py --dataset Beauty

The ground truth matching for those evaluation axes again leverages pre-trained Sentence-T5 models.
Finally, if you would like to run our Mender-Emb variant, you will need to embed the matched instructions and the different data splits by

    python embed_items.py --embedding_model google/flan-t5-small
    python embed_preferences.py --embedding_model google/flan-t5-small --dataset Beauty
    python embed_preferences.py --accumulate --embedding_model google/flan-t5-small --dataset Beauty
    python embed_fine_coarse_preferences.py --embedding_model google/flan-t5-small
    python embed_pos_neg_preferences.py --embedding_model google/flan-t5-small --dataset Beauty

There are various embedding models supported, namely the FLAN-T5 series, Instructor models, BLAIR, and GritLM.

### Training Multimodal Generative Retrieval Models

To train our Mender-Tok model on the Amazon Beauty subset execute the following command.

``` bash
python run.py dataset=MenderTok_Beauty
```

This will automatically download data for Beauty and train the [RQ-VAE](https://arxiv.org/abs/2203.01941) for generation of semantic ids.
All results related to this run will be dumped to the logging directory specified by hydra and the wandb run will also be initialized based on the name of this directory.
To train on the yelp dataset, you will need to download the data from [here](https://www.yelp.com/dataset/download) first.
The data for steam will also be automatically downloaded upon training on the Steam dataset.

We provide support for training all generative retrieval models as reported in the paper in this codebase.
The different model configurations can be found in the configs directory.
For running the Mender-Emb variant, for example, you will need to execute

``` bash
python run.py dataset=MenderEmb_Beauty
```

### Results

We provide all our results for the different methods in the ```results``` directory.
This directory follows a certain structure to create the starplots and the barplots for the main experiments and ablation studies as shown in the paper.
The re-generate plots for the main results on the five dataset for example, you will need navigate to the ```postprocessing``` directory and execute

    python starplot_comparison.py --ckpt_path ../results/final

This will load the ```result_dict.pkl``` for each seed for each method and average them and plot them on the starplot.

### Citation

If you found our work useful, please consider citing it

    @misc{paischer_prefdisc_2024,
      title={Preference Discerning with LLM-Enhanced Generative Retrieval},
      author={Fabian Paischer and Liu Yang and Linfeng Liu and Shuai Shao and Kaveh Hassani and Jiacheng Li and Ricky Chen and Zhang Gabriel Li and Xialo Gao and Wei Shao and Xue Feng and Nima Noorshams and Sem Park and Bo Long and Hamid Eghbalzadeh},
      year={2024},
      eprint={2412.08604},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.08604},
    }

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

The majority of "Preference Discerning with LLM-Enhanced Generative Retrieval" is licensed under CC-BY-NC , as found in the [LICENSE](LICENSE) file., however portions of the project are available under separate license terms: rqvae is licensed Apache 2.0.
