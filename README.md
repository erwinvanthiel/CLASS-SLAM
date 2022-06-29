# Champagne Taste on a Beer Budget: Better Budget Utilisation in Multi-label Adversarial Attacks

# Abstract
Abstract—Multi-label classification is an important branch of
classification as in many real world classification scenarios the
subject belongs to multiple classes simultaneously. In single-label
scenarios the adversarial attacks conventionally are constrained
by a perturbation budget in order to enforce imperceptibility. In
the related work concerning multi-label attacks there has been
no notion of a budget, which results in obvious perturbations.
In this paper we try to achieve the most possible flips for a
given budget. We start out this research by investigating the
applicability of existing optimization-based single label attack
MI-FGSM straight-out-of-the box to a multiple label scenario.
It becomes apparent that this method distributes the budget
over all labels simultaneously while it does not have sufficient
budget, hence we say it has champagne taste on a beer budget.
We find that all labels have different attackability and exhibit
different correlation structures. We also examine the effect of
the loss function for this optimisation problem and how it
influences label prioritisation. We use this knowledge to design
two distinct methods namely, Classification Landscape Attentive
Subset Selection(CLASS) and Smart Loss-function for Attacks on
Multi-label models (SLAM). CLASS comprises a suitable subset
of labels we will distribute our budget over, while considering
the labels their attackability and pairwise correlation. SLAM
comprises a loss function that adapts to the budget and the
attacked classifier. We extensively evaluate CLASS and SLAM
on three data sets, against two state of the art models, namely
Query2Label and ASL. Our evaluation results show that CLASS
and SLAM are able to increase the flips given the budget
constraint by up to 131% and 61% respectively.
Index Terms—Deep Learning, Multi-label Classification, Ad-
versarial Attacks,

## Run instructions:

* Use requirements.txt to install the required packages.
* In the .json config files of the model(s) (example: asl_coco.json) change the resume/model_path variable to the location ()of the model paramters (example: ../../models/tresnetxl-asl-voc-epoch80).
* Make sure there is a folder called experiment_results in the repo root directory.
* Then enter the following command: python script_name.py classifier path_to_dataset dataset_type
	* example: python attack.py asl_nuswide ../../NUS_WIDE NUS_WIDE
	* For other options such as batch_size, threshold, etc, run without arguements for help
	* Results are stored as Numpy files and can be printed with scripts in plots folder

