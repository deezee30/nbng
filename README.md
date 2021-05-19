# No Brain No Gain

No Brain No Gain is a modular collection of neural activity decoding algorithms designed for BIOE97075 - Brain Machine Interfaces 2020-2021 @ Imperial College London.

## Installation

No installation required.

## Usage

The source code comes with five decoding models (located in the [models folder](src/models)):
- `bayes_avjtraj` - Bayes Classification w/ Average Trajectory Prediction ([link](src/models/bayes_avjtraj))
- `knn_linreg` - k-Nearest Neighbours Classification w/ Linear Regression ([link](src/models/knn_linreg))
- `svm_28` - 28 Support Vector Machines ([link](src/models/svm_28))
- **`svm_4` - 4 Support Vector Machines ([link](src/models/svm_4)) [Chosen Design]**
- `wknn_avjtraj` - Weighted k-Nearest Neighbours Classification w/ Average Trajectory Prediction ([link](src/models/wknn_avjtraj))

To use, configure the following settings in `main.m`. For multiple runs (with or without varying parameters), extend the `seeds` and `data_splits` vectors into more than one element:

```
model       = "svm_4"; % model name (corresponding to folder)
seeds       = [2013];  % seeds for random permutations
data_splits = [.8];    % cross validation training/testing ratios: 0.8 -> 80/20
```

Next, run `main.m` or call the `decoder` function directly.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

All code licensed under [MIT License](https://choosealicense.com/licenses/mit/). See [LICENSE.txt](LICENSE.txt) for further details.