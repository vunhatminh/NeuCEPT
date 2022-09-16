#!/bin/bash
for label in 0 1
do
	for mode in 'base2' 'distill'
	do
		python VGG_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift GradientSHAP --top 10 --layer 12 --label $label 
		python VGG_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift GradientSHAP --top 20 --layer 12 --label $label 
		python VGG_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift GradientSHAP --top 30 --layer 12 --label $label 

		python VGG_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift GradientSHAP --top 20 --layer 13 --label $label 
		python VGG_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift GradientSHAP --top 50 --layer 13 --label $label 
		python VGG_clustering.py --mode $mode --explainer FDR Saliency IntegratedGradients DeepLift GradientSHAP --top 100 --layer 13 --label $label 
	done
done