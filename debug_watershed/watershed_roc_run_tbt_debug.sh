#!/bin/bash -l

#SBATCH
#SBATCH --time=24:00:00
#SBATCH --partition=lrgmem
#SBATCH --nodes=1




unsupervised_learning_input_dir="$1"
watershed_run_dir="$2"
pseudocount="$3"
pvalue_fraction="$4"
gradient_descent_threshold="$5"
theta_pair_init="$6"
lambda_pair="$7"

# Parameters!
number_of_dimensions="49"

gene_thresh="0.01"
input_stem="total_expression_tbt_outliers_"$gene_thresh"_genes_intersection_between_te_ase_splicing"
input_file=$unsupervised_learning_input_dir$input_stem"_features_filter_N2_pairs.txt"
output_stem=$watershed_run_dir"total_expression_imp_tbt_outliers_intersection_te_ase_splicing_outliers_gene_pvalue_"$gene_thresh"_outlier_fraction_"$pvalue_fraction"_pseudocount_"$pseudocount"_gradient_thresh_"$gradient_descent_threshold"_theta_pair_init_"$theta_pair_init"_lambda_pair_"$lambda_pair"_log_reg_init"

echo $pseudocount
echo $pvalue_fraction
echo $gene_thresh
echo $gradient_descent_threshold
echo $theta_pair_init
echo $lambda_pair

Rscript watershed_roc_tbt_debug.R $pvalue_fraction $input_file $output_stem $number_of_dimensions $pseudocount $gradient_descent_threshold $theta_pair_init $lambda_pair
