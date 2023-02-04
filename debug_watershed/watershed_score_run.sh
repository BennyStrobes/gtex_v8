#!/bin/bash -l

#SBATCH
#SBATCH --time=24:00:00
#SBATCH --mem=60GB
#SBATCH --partition=lrgmem
#SBATCH --nodes=1




unsupervised_learning_input_dir="$1"
watershed_run_dir="$2"
pseudocount="$3"
pvalue_fraction="$4"

# Parameters!
number_of_dimensions="3"
inference_method="exact"

gene_thresh="0.01"
input_stem="fully_observed_merged_outliers_"$gene_thresh"_genes_intersection_between_te_ase_splicing"
fully_observed_input_file=$unsupervised_learning_input_dir$input_stem"_features_filter_N2_pairs.txt"

all_variants_input_file=$unsupervised_learning_input_dir"all_availibile_samples_features_filter_partially_observed_expression.txt"
all_variants_variant_level_input_file=$unsupervised_learning_input_dir"all_availibile_samples_variant_level_features_filter_partially_observed_expression.txt"

output_stem=$watershed_run_dir"fully_observed_te_ase_splicing_outliers_gene_pvalue_"$gene_thresh"_outlier_fraction_"$pvalue_fraction"_pseudocount_"$pseudocount"_"$inference_method"_inference_apply_to_all_variants"


Rscript watershed_score.R $pvalue_fraction $number_of_dimensions $inference_method $pseudocount $fully_observed_input_file $all_variants_input_file $all_variants_variant_level_input_file $output_stem