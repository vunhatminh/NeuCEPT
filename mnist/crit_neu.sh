#!/bin/bash
## EXPLAINING INPUT

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    python explain_LeNet.py --explainer Saliency --label $label
    python explain_LeNet.py --explainer IntegratedGradients --label $label
    python explain_LeNet.py --explainer DeepLift --label $label
    python explain_LeNet.py --explainer SmoothGrad --label $label
    python explain_LeNet.py --explainer GradientSHAP --label $label
    python explain_LeNet.py --explainer GuidedGradCam --label $label
    python explain_LeNet.py --explainer Lime --label $label
    python fdr_LeNet.py --explainer FDR --label $label --fdr 0.1 --runs 100
done

## EXPLAINING LAYER

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for layer in 'linear0' 'conv3'
    do
        python explain_LeNet_Layer.py --explainer Saliency --label $label --layer $layer
        python explain_LeNet_Layer.py --explainer IntegratedGradients --label $label --layer $layer
        python explain_LeNet_Layer.py --explainer DeepLift --label $label --layer $layer
        python explain_LeNet_Layer.py --explainer SmoothGrad --label $label --layer $layer
        python explain_LeNet_Layer.py --explainer GradientSHAP --label $label --layer $layer
        python fdr_LeNet.py --explainer FDR --label $label --fdr 0.02 --runs 50 --layer $layer
    done
done

## EXPLAINING INPUT
for mode in 'mod2' 'mod5' 'distillmod2' 'distillmod5'
do
    python explain_LeNet.py --mode $mode --explainer Saliency --label 0
    python explain_LeNet.py --mode $mode --explainer IntegratedGradients --label 0
    python explain_LeNet.py --mode $mode --explainer DeepLift --label 0
    python explain_LeNet.py --mode $mode --explainer SmoothGrad --label 0
    python explain_LeNet.py --mode $mode --explainer GradientSHAP --label 0
    python explain_LeNet.py --mode $mode --explainer GuidedGradCam --label 0
    python explain_LeNet.py --mode $mode --explainer Lime --label 0
    python fdr_LeNet.py --mode $mode --explainer FDR --label 0 --fdr 0.1 --runs 100
done

## EXPLAINING LAYER
for mode in 'mod2' 'mod5' 'distillmod2' 'distillmod5'
do
    for layer in 'linear0' 'conv3'
    do
        python explain_LeNet_Layer.py --mode $mode --explainer Saliency --label 0 --layer $layer
        python explain_LeNet_Layer.py --mode $mode --explainer IntegratedGradients --label 0 --layer $layer
        python explain_LeNet_Layer.py --mode $mode --explainer DeepLift --label 0 --layer $layer
        python explain_LeNet_Layer.py --mode $mode --explainer SmoothGrad --label 0 --layer $layer
        python explain_LeNet_Layer.py --mode $mode --explainer GradientSHAP --label 0 --layer $layer
        python fdr_LeNet.py --mode $mode --explainer FDR --label 0 --fdr 0.02 --runs 50 --layer $layer
    done
done
