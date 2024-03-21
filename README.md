# C$^2$KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation 
Code for paper "C$^2$KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation".

## Usage

### Requirements
requirements.txt

### Data Preparation
Download Original Dataset：
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
[AVE](https://sites.google.com/view/audiovisualresearch),
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/),


### Pre-processing

For AVE, CREMA-D and VGGSound dataset, we provide code to pre-process videos into RGB frames and audio wav files in the directory ```utils/data/```.

## Run commands
Detailed descriptions of options can be found in [main_overlap_tag.py](main_overlap_tag.py)
1. Pre-train the single modality model
2. Conduct crossmodal knowledge distillation