# Run code

### Run this commnand to show all information about running code

    python run.py -h

# References

This repo implements the model in the following paper:

- Twitter User Geolocation using Deep Multiview
Learning
    - Tien Huu Do, Duc Minh Nguyen, Evaggelia Tsiligianni, Bruno Cornelis, Nikos Deligiannis

# Requirements
- pandas
- tensorflow 2.x
- numpy
- matplotlib
- seaborn
- pyvi
- gensimpandas
- scikit-learn
- requests
- python 3.x

# Configuration

The below parameters can be adjusted to figure out the best value of each parameters. 
Parameter | Required    | Default
---       | ---         | ---
`-h`, `--help`  |`False`  |
`-f <bool>`, `--from-scratch <bool>` | `True` |
`-ct <bool>`, `--context-only <bool>` | `False` | `False`
`-p <bool>`, `--process-input <bool>`   | `False`   |`False`
`d <bool>`, `--docvec-pretrain <bool>`  |`False` |`False`
`-lr <float>`, `--learning-rate <float>`    |`False`  |0.01
`o <string>`, `--optimizer <string>`  |`False`    |`SGD`
`--epochs <int>`    |`False`    |`10`

# Demo

    python app.py

- Input: facebook link or ID of a user in `dataset/textdata/data.csv`

# Data
<a href="https://drive.google.com/drive/folders/1esqFOFInSjinRrdEnRljCSyo842cGOuV?usp=sharing" target="_blank">Google drive link</a>

Folder must be in the same directory as `run.py` for the program be able to run

# Split train test data

    python utils.py