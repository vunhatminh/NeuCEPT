#!/bin/bash

for label in 0 1 2 3 4 5 6 7 8 9
do
 for layer in 12 13
 do
   python explain_VGG_layer.py --explainer Saliency --label $label --layer $layer
   python explain_VGG_layer.py --explainer IntegratedGradients --label $label --layer $layer
   python explain_VGG_layer.py --explainer DeepLift --label $label --layer $layer
   python explain_VGG_layer.py --explainer GradientSHAP --label $label --layer $layer
   python fdr.py --label $label --runs 50 --layer $layer
 done
done

for label in 0 1
do
    python fdr.py --mode base2 --label $label --runs 50 --layer 12
    python fdr.py --mode distill --label $label --runs 50 --layer 12
    python fdr.py --mode base2 --label $label --runs 50 --layer 13
    python fdr.py --mode distill --label $label --runs 50 --layer 13
done

for label in 0 1
do
   python explain_VGG_layer.py --mode base2 --explainer IntegratedGradients --label $label --layer 12
   python explain_VGG_layer.py --mode base2 --explainer GradientSHAP --label $label --layer 12
   python explain_VGG_layer.py --mode base2 --explainer DeepLift --label $label --layer 12 
   python explain_VGG_layer.py --mode base2 --explainer Saliency --label $label --layer 12

   python explain_VGG_layer.py --mode base2 --explainer IntegratedGradients --label $label --layer 13
   python explain_VGG_layer.py --mode base2 --explainer GradientSHAP --label $label --layer 13
   python explain_VGG_layer.py --mode base2 --explainer DeepLift --label $label --layer 13
   python explain_VGG_layer.py --mode base2 --explainer Saliency --label $label --layer 13
done

for label in 0 1
do
   python explain_VGG_layer.py --mode distill  --explainer IntegratedGradients --label $label --layer 12 
   python explain_VGG_layer.py --mode distill  --explainer GradientSHAP --label $label --layer 12 
   python explain_VGG_layer.py --mode distill  --explainer DeepLift --label $label --layer 12 
   python explain_VGG_layer.py --mode distill  --explainer Saliency --label $label --layer 12 

   python explain_VGG_layer.py --mode distill  --explainer IntegratedGradients --label $label --layer 13 
   python explain_VGG_layer.py --mode distill  --explainer GradientSHAP --label $label --layer 13 
   python explain_VGG_layer.py --mode distill  --explainer DeepLift --label $label --layer 13 
   python explain_VGG_layer.py --mode distill  --explainer Saliency --label $label --layer 13
done
