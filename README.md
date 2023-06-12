# Seizure detection from EEG data
This repo contains the main training script used in our work "Towards trustworthy seizure onset detection using workflow notes" [[arXiv](https://github.com/khaledsaab/eeg_robustness)].


## Requirements

pytorch, torchvision, torchaudio, pytorch-lightning, transformers, hydra-core, rich, wandb, hydra_colorlog (pip), eeghdf=0.1 (pip), einops, opt_einsum, scipy (pip, for s4 model)

## Training run


## Codebase Structure

Notation <br>
`├── folders` <br>
`|=== files`<br>

```bash
    |=== train.py                              # (MAIN) running the model
    ├── configs                    
        |=== config.yaml                       # default config file 
    ├── src
        ├── models                             # contains model architectures
        ├── callbacks                          
        ├── optim                           
        ├── tasks                           
        ├── datamodules                        # directory for datamodules                 
            |=== eeg_datamodule.py             # defines EEG dataloaders
            |=== eeg_utils.py                  # utility functions for EEG data
    ├── datasets                               # directory where datasets live
    └── README.md
```



## Acknowledgements
`csrc` directory copied from `https://github.com/HazyResearch/zoo`

Code copied, adapted and inspired from the following repositories:
- `https://github.com/HazyResearch/zoo`
- `https://github.com/HazyResearch/state-spaces`
- `https://github.com/tsy935/eeg-gnn-ssl`


## Citation

If you use this codebase, or otherwise found our work valuable, please cite: