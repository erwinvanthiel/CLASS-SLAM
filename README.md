# Champagne Taste on a Beer Budget: Better Budget Utilisation in Multi-label Adversarial Attacks

# Abstract
Multi-label classification is an important branch
of classification problems as in many real world classification
scenarios the subject belongs to multiple classes simultaneously.
The classifiers have been shown to be vulnerable against small in-
put perturbations, called adversarial examples. There are multi-
class classifiers, in which data instances get assigned to a single
class and multi-label classifiers, in which a data instance can
be associated with multiple labels simultaneously. In multi-class
scenarios these adversarial attacks are conventionally constrained
by a perturbation magnitude budget in order to enforce visual
imperceptibility. In the related studies concerning multi-label
attacks there has been no notion of a budget, which results in
clearly visible perturbations in the image.
In this paper we develop attacks that cause the most severe
disruptions in the binary label predictions, i.e. label flips, within
a budget. To achieve this, we first conduct an empirical analysis
to answer the applicability of the exiting single label attacks
on multiple label problems. Our key observations are that
targeting all labels simultaneously in small budgets leads to sub-
optimal results, that all labels have different attackability and
also that labels exhibit different correlation structures which
influences their combined attackability. Moreover, we show that
the loss function determines the prioritisation among labels
during optimisation. We conclude that there are two methods
that allow for more efficient budget utilisation. We can construct
our loss function such that it ensures a prioritisation that is
too greedy nor too patient for the budget. We can also put our
focus on a subset of labels as opposed to targeting them all
simultaneously.
Because of this we design two distinct methods namely,
Classification Landscape Attentive Subset Selection(CLASS) and
Smart Loss-function for Attacks on Multi-label models (SLAM).
CLASS comprises a subset of labels we will distribute our budget
over, which was constructed while considering the labels their
attackability and pairwise correlation. SLAM comprises a loss
function that uses an estimate for the potential amount of flips to
adapt the shape of its curve, and hence the label prioritisation.
We extensively evaluate CLASS and SLAM on three data sets,
against two state of the art models, namely Query2Label and
ASL. Our evaluation results show that CLASS and SLAM are
able to increase the flips given the budget constraint by up to
131% and 61% respectively.

## Run instructions:

* Use requirements.txt to install the required packages.
* In the .json config files of the model(s) (example: asl_coco.json) change the resume/model_path variable to the location ()of the model paramters (example: ../../models/tresnetxl-asl-voc-epoch80).
* Make sure there is a folder called experiment_results in the repo root directory.
* Then enter the following command: python script_name.py classifier path_to_dataset dataset_type (NUS_WIDE / MSCOCO_2014 / VOC2007)
	* example: python attack.py asl_nuswide ../../NUS_WIDE NUS_WIDE
	* For other options such as batch_size, threshold, etc, run without arguements for help
	* Results are stored as Numpy files and can be printed with scripts in plots folder

