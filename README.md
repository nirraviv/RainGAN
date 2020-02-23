# RainGAN
Applying simGAN for generating rainfall maps

## Requirements

This repo tested on Python 3.6 and PyTorch 1.4.

The rest of the libraries are in requirements.txt

## Dataset
The real dataset was downloaded from SMHI website (https://opendata.smhi.se/apidocs/radar/)

### Real images dataset
- Download to tiff images: python utils/dowload_radar_images.py (-h for parser arguments)
- Preprocess: python utils/cut_radar_images.py (-h for parser arguments)

### Synthetic images dataset
- Run: python utils/random_gaussian_generation.py (-h for parser arguments)
## GAN
All configuration parameters to train and evaluate RainGAN are in config.py
### Train
- In configuration file set __is_train=True__ and __is_eyes=False__
- Run: python main.py

### Evaluate
- In configuration file set __load_from_iter__, __is_train=False__ and __is_eyes=False__
- Run: python main.py

