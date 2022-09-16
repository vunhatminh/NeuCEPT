#!/bin/bash

for label in 0 1 2 3 4 5 6 7 8 9
do
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 100 --noise_step 5 --gamma 10 --top 5 --layer 12
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 100 --noise_step 5 --gamma 10 --top 10 --layer 12
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 100 --noise_step 5 --gamma 100 --top 5 --layer 12 
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 100 --noise_step 5 --gamma 100 --top 10 --layer 12 

	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 40000 --noise_step 2000 --gamma 10 --top 5 --layer 13
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 40000 --noise_step 2000 --gamma 10 --top 10 --layer 13 
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 40000 --noise_step 2000 --gamma 100 --top 5 --layer 13 
	python ablation_VGG_layer.py --explainer FDR_base10 Saliency IntegratedGradients DeepLift GradientSHAP --label $label --noise_max 40000 --noise_step 2000 --gamma 100 --top 10 --layer 13
done