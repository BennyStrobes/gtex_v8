#!/bin/bash -l

#SBATCH
#SBATCH --time=7:00:00
#SBATCH --partition=lrgmem

#SBATCH --nodes=1

######## Ben Strober
######## 11/27/18


#######################################
# Input Data (Assumes "outlier_calling" (https://github.com/BennyStrobes/gtex_v8_rare_splice/tree/master/outlier_calling) and enrichment analysis (https://github.com/BennyStrobes/gtex_v8_rare_splice/tree/master/enrichment_analysis) has been already run)
#######################################


# List of european ancestry individuals (extracted from Nicole Ferraro's variant list: "all_rare_variants_SNPs_10kb_genebody.txt")
european_ancestry_individual_list="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/variant_calls/european_ancestry_individuals.txt"

# File containing genomic annotations describing rare variants
# Downloaded from stroberb-420@scp.nygenome.org:/data/delivery/gtex-rare/data/gtex_v8_vep88_loftee_cadd_gnomad.tsv.gz on 1/9/19
raw_genomic_annotation_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/variant_calls/gtex_v8_rare_indivgeno_vep88_loftee_cadd_gnomad.tsv"

# File containing mapping of rare variants to genes
rare_variant_to_gene_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/variant_calls/all_rare_variants_SNPs_10kb_genebody.txt"

# File containing all rare variants
variant_bed_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/variant_calls/all_rare_variants_noWindow_SNPs.txt"

# Total expression outlier file
total_expression_outlier_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/outlier_calls/gtexV8.outlier.controls.v8ciseQTLs.globalOutliers.removed.medz.txt"

# Allele specific expression outlier file
ase_outlier_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/outlier_calls/median_uncorrected_DOT_scores.tsv"

# Splicing outlier file
splicing_outlier_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/outlier_calling/splicing_outlier_calls/cross_tissue_covariate_method_none_no_global_outliers_ea_only_emperical_pvalue_gene_level.txt"

# Directory containing processed genomic annotations
genomic_annotation_dir="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/genomic_annotations/"

# File containing genomic annotations
genomic_annotation_file=$genomic_annotation_dir"filtered_real_valued_gene_level_variant_compressed_genomic_annotations.txt"

# File containing genomic annotations for each variant
variant_level_genomic_annotation_file=$genomic_annotation_dir"filtered_real_valued_gene_level_clean_genomic_annotations.txt"

# File containing mapping from (gene, individual) to rare variants mapped within 10KB of the gene
gene_individual_to_variant_mapping_file=$genomic_annotation_dir"variants_compressed_onto_genes.txt"

# Ordered list of GTEx v8 tissue names
tissue_names_file="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/input_data/gtex_v8_tissues.txt"

# Directory containing splicing outlier calls
splicing_outlier_dir="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/outlier_calling/splicing_outlier_calls/"

#############################################################
#Used Directories (directories need to be created and empty before starting)
#############################################################

# Root directories that all other directories will be placed in 
output_root="/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/unsupervised_modeling/"


# Directory containing unsupervised learning input files
unsupervised_learning_input_dir=$output_root"unsupervised_learning_input/"

# Directory containing results from single outlier type RIVER runs
river_run_dir=$output_root"river/"

# Directory containing results from watershed ROC analysis
watershed_3_class_roc_run_dir=$output_root"watershed_three_class_roc/"

# Directory containing results from watershed tbt ROC analysis
watershed_tbt_roc_run_dir=$output_root"watershed_tbt_roc/"

# Directory containing results from watershed analysis applied to all variants
watershed_3_class_score_run_dir=$output_root"watershed_three_class_scores/"



###############################################
# Scripts 
###############################################
if false; then
sh prepare_input_files_for_unsupervised_learning_methods.sh $genomic_annotation_file $variant_level_genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir $gene_individual_to_variant_mapping_file $splicing_outlier_dir $tissue_names_file
fi

if false; then
sh river_copy_run.sh $unsupervised_learning_input_dir $river_run_dir
fi

pseudocount="30"
pvalue_fraction=".01"
gradient_descent_threshold=".005"
theta_pair_init="4"
if false; then

sh watershed_roc_run_3_outlier_types.sh $unsupervised_learning_input_dir $watershed_3_class_roc_run_dir $pseudocount $pvalue_fraction $gradient_descent_threshold $theta_pair_init

pseudocount="30"
pvalue_fraction=".02"
sh watershed_roc_run_3_outlier_types.sh $unsupervised_learning_input_dir $watershed_3_class_roc_run_dir $pseudocount $pvalue_fraction $gradient_descent_threshold $theta_pair_init

pseudocount="30"
pvalue_fraction=".03"
sbatch watershed_roc_run_3_outlier_types.sh $unsupervised_learning_input_dir $watershed_3_class_roc_run_dir $pseudocount $pvalue_fraction $gradient_descent_threshold $theta_pair_init

pseudocount="30"
pvalue_fraction=".04"
sbatch watershed_roc_run_3_outlier_types.sh $unsupervised_learning_input_dir $watershed_3_class_roc_run_dir $pseudocount $pvalue_fraction $gradient_descent_threshold $theta_pair_init

pseudocount="30"
pvalue_fraction=".05"
sbatch watershed_roc_run_3_outlier_types.sh $unsupervised_learning_input_dir $watershed_3_class_roc_run_dir $pseudocount $pvalue_fraction $gradient_descent_threshold $theta_pair_init


pseudocount="30"
pvalue_fraction=".01"
sbatch watershed_score_run.sh $unsupervised_learning_input_dir $watershed_3_class_score_run_dir $pseudocount $pvalue_fraction
fi



#####################
# TBT Model
# running.0005 should do .0001
#####################
#Rscript temp_opti.R

pseudocount=".001"
pvalue_fraction=".01"
theta_pair_init="0"
lambda="0.01"
lambda_pair="0"
if false; then
sbatch watershed_roc_run_tbt.sh $unsupervised_learning_input_dir $watershed_tbt_roc_run_dir $pseudocount $pvalue_fraction $theta_pair_init $lambda $lambda_pair
fi


step_size=".8"
convergence_thresh=".00001"
sh temp_opti.sh $step_size $convergence_thresh





