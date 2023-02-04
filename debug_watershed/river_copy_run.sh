#!/bin/bash -l

#SBATCH
#SBATCH --time=16:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1


unsupervised_learning_input_dir="$1"
river_run_dir="$2"




#############################################
# Comparison between TE and splicing
#############################################
gene_thresh="0.01"
pvalue_thresholds=( ".1" ".05" ".01" ".005" ".001")

for pvalue_threshold in "${pvalue_thresholds[@]}"; do
	echo $pvalue_threshold
	## TOTAL EXPRESSION
	# Name of test (used to save files)
	stem="fully_observed_total_expression_outliers_"$gene_thresh"_genes_intersection_between_te_splicing"
	# File containing input data
	input_file=$unsupervised_learning_input_dir$stem"_features_filter_N2_pairs.txt"
	Rscript evaRIVER_copy.R $pvalue_threshold $input_file $stem $river_run_dir
	## SPLICING
	# Name of test (used to save files)
	stem="fully_observed_splicing_outliers_"$gene_thresh"_genes_intersection_between_te_splicing"
	# File containing input data
	input_file=$unsupervised_learning_input_dir$stem"_features_filter_N2_pairs.txt"
	Rscript evaRIVER_copy.R $pvalue_threshold $input_file $stem $river_run_dir
done

gene_thresh="0.05"
pvalue_thresholds=( ".1" ".05" ".01" ".005" ".001")
for pvalue_threshold in "${pvalue_thresholds[@]}"; do
	echo $pvalue_threshold
	## TOTAL EXPRESSION
	# Name of test (used to save files)
	stem="fully_observed_total_expression_outliers_"$gene_thresh"_genes_intersection_between_te_splicing"
	# File containing input data
	input_file=$unsupervised_learning_input_dir$stem"_features_filter_N2_pairs.txt"
	Rscript evaRIVER_copy.R $pvalue_threshold $input_file $stem $river_run_dir
	## SPLICING
	# Name of test (used to save files)
	stem="fully_observed_splicing_outliers_"$gene_thresh"_genes_intersection_between_te_splicing"
	# File containing input data
	input_file=$unsupervised_learning_input_dir$stem"_features_filter_N2_pairs.txt"
	Rscript evaRIVER_copy.R $pvalue_threshold $input_file $stem $river_run_dir
done