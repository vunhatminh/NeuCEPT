#!/bin/bash
## UNSUPERVISED LAYER
for mode in 'mod2' 'mod5' 'distillmod2' 'distillmod5'
do
    for layer in 'linear0' 'conv3'
    do
        python neuron_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift  GradientSHAP --top 5 --layer $layer
        python neuron_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift  GradientSHAP --top 10 --layer $layer
        python neuron_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift  GradientSHAP --top 20 --layer $layer
    done
done
