# [NDT3 Model Card](https://github.com/joel99/ndt3/blob/main/model_card.md)

We report essential context on NDT3 models here, adapted from [Model Cards for Model Reporting (Mitchell et al.), 2024](https://arxiv.org/abs/2404.07202).

## Model Details

NDT3 is a family of autoregressive Transformers that model the neural spiking activity and low-dimensional behavioral data. It was primarily trained on data from the [Cortical Bionics Research Group](https://www.corticalbionics.com/) to evaluate the efficacy of large scale pretraining for motor decoding.

### Date
September 2024

### Model type
Autoregresive Transformer over Neural Data and Low Dimensional Behavior

### Model version
Up to 350M parameters

### Paper
TODO.

### Questions and Feedback
Contact: [joelye9@gmail.com](mailto:joelye9@gmail.com).

## Intended Model Use
This model is intended to help accelerate data collection for neuroscience and iBCI communities. It may also be of interest to the machine learning community for investigating properties of transfer learning across neural datasets.

## Data

### Modalities
- Neural data: Microelectrode recordings
- Behavior: Low-dimensional behavior either from:
    - Keypoint tracking (arm, hand)
    - Manipulandum sensors (force)
    - Virtual state (cursor position, MuJoCo simulator state)

### Model systems

- Humans (Spinal Cord Injury, enrolled in a clinical trial for sensorimotor neuroprosthetics)
- Macaque monkeys

### Task resolution
- Token/sequence: single motions.
- Pretraining: Reach and Grasp (arm, wrist, hand, fingers), Brain-machine interface control of a robotic arm and computer cursor

### Spatial coverage
- Token: 32 Channels of single or multi-unit spiking activity.
- Sequence: <320 Channels of single or multi-unit spiking activity.
- Pretraining: Motor cortex (primary and premotor)

### Temporal breadth and resolution
- Token: 20ms binned spike counts.
- Sequence: 2s timescale behavior.
- Pretraining: Up to 2500 hours.

### Preprocessing
- Neural spiking was extracted from 30kHz voltage recordings and variously filtered and thresholded to yield spikes. Some datasets further sort activity to attribute spikes to different putative neurons.
- Behavior data, when known to be positional, was converted to velocity.
- Human open loop behavior was smoothed using a causal 300ms rolling average.
- Behavior data was normalized to have a max amplitude of 1, and sometimes mean-centered.

### Pretraining data

The majority of data was extracted from archived experiments from several different labs, including labs affiliated with the [Cortical Bionics Research Group](https://www.corticalbionics.com/), the Schwartz, Chase and SMILE (Batista) labs, and the Precision Neural Dynamics Lab (Adam Rouse). We also include a few public datasets from the Churchland lab and Flint lab. Most public datasets were reserved for evaluation. Some stopwords were used to exclude poor quality data, but minimal automatic filtering was performed overall.

### Evaluation Data
- [FALCON](https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1) H1, M1, M2. (Karpowicz et al, 2024)
- [Random target reaching](https://zenodo.org/records/583331) (O'Doherty et al, 17)
- Private data: Human open-loop grasp, 2D cursor + click; Monkey center-out, [critical stability task](https://pubmed.ncbi.nlm.nih.gov/29947593/) (Quick et al, 18), [oculomotor pursuit](https://doi.org/10.1145/3649902.3655655) (Noneman and Mayo, 24).

### Performance, and Limitations
NDT3 improves decoding on downstream tasks over from-scratch training when downstream data volumes is approximately under an hour. However, we find limited scaling when increaisng pretraining data.

### Ethical Considerations
NDT3 uses and evaluates data from implanted devices. Such devices are essential for research for basic neuroscience and rehabilitative neurotechnology for restoring function to individuals with impairments resulting from brain injury or disease. Pretrained models may improve the efficiency of downstream data collection; we are thus releasing model weights for study and re-use. As the model only receives binned spiking activity, we believe there is minimal risk to human participant privacy, but ask users to not attempt to reverse engineer the model to recreate the training data.