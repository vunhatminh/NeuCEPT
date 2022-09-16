# LeNet model on MNIST

Make sure that the directories `data/`, `pretrained/`, `result/` exist before executing the code.

Grant execution permission for bash scripts:

```bash
$ chmod +x *.sh
```

## Train models
Execute the following command:

```bash
$ ./train.sh
```

The trained models will be saved in `pretrained/`.

## Extract critical neurons
Execute the following command:

```bash
$ ./crit_neu.sh
```

The outputting score of layers' neurons will be saved as `.npy` files in `result/`.

## Ablation test
Execute the following command:

```bash
$ ./ablation.sh
```

The result will be saved as `Ablation_*.pkl` files in `result/`.

## Clustering
Execute the following command:

```bash
$ ./clustering.sh
```
The result will be saved as `?(mod2|mod5|distillmod2|distillmod5)_ClusMethod*.pkl` files in `result/`.
