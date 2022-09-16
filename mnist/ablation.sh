#!/bin/bash

## ABLATION INPUT CONTINUOUS

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    python ablation_LeNet.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP GuidedGradCam Lime --label $label --noise_max 500 --noise_step 10 --gamma 10
    python ablation_LeNet.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP GuidedGradCam Lime --label $label --noise_max 500 --noise_step 10 --gamma 20
done

## ABLATION INPUT DISCRETE TOP K

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    python ablation_LeNet.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP GuidedGradCam Lime --label $label --noise_max 500 --noise_step 10 --gamma 10 --top 100
    python ablation_LeNet.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP GuidedGradCam Lime --label $label --noise_max 500 --noise_step 10 --gamma 20 --top 100
    python ablation_LeNet.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP GuidedGradCam Lime --label $label --noise_max 500 --noise_step 10 --gamma 100 --top 100
done

## ABLATION LAYER DISCRETE TOP K

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for layer in 'linear0' 'conv3'
    do
        python ablation_LeNet_Layer.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP --label $label --noise_max 40000 --noise_step 1000 --gamma 10 --top 5 --layer $layer
        python ablation_LeNet_Layer.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP --label $label --noise_max 40000 --noise_step 1000 --gamma 10 --top 10 --layer $layer
        python ablation_LeNet_Layer.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP --label $label --noise_max 40000 --noise_step 1000 --gamma 100 --top 5 --layer $layer
        python ablation_LeNet_Layer.py --explainer FDR Saliency IntegratedGradients DeepLift  SmoothGrad GradientSHAP --label $label --noise_max 40000 --noise_step 1000 --gamma 100 --top 10 --layer $layer
    done
done
