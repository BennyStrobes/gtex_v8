#!/bin/bash -l

#SBATCH
#SBATCH --time=5:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1


genomic_annotation_file="$1"
variant_level_genomic_annotation_file="$2"
total_expression_outlier_file="$3"
ase_outlier_file="$4"
splicing_outlier_file="$5"
unsupervised_learning_input_dir="$6"
gene_individual_to_variant_mapping_file="$7"
splicing_outlier_dir="$8"
tissue_names_file="$9"


pvalue=".01"
echo $pvalue
if false; then

python prepare_input_files_for_unsupervised_learning_intersection_te_ase_splicing.py $genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir $pvalue $gene_individual_to_variant_mapping_file

python prepare_input_files_for_tbt_splicing.py $unsupervised_learning_input_dir $pvalue $splicing_outlier_dir $tissue_names_file
fi


if false; then
pvalue=".02"
echo $pvalue
python prepare_input_files_for_unsupervised_learning_intersection_te_ase_splicing.py $genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir $pvalue $gene_individual_to_variant_mapping_file


pvalue=".03"
echo $pvalue
python prepare_input_files_for_unsupervised_learning_intersection_te_ase_splicing.py $genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir $pvalue $gene_individual_to_variant_mapping_file

pvalue=".04"
echo $pvalue
python prepare_input_files_for_unsupervised_learning_intersection_te_ase_splicing.py $genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir $pvalue $gene_individual_to_variant_mapping_file


pvalue=".05"
echo $pvalue
python prepare_input_files_for_unsupervised_learning_intersection_te_ase_splicing.py $genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir $pvalue $gene_individual_to_variant_mapping_file


python prepare_input_files_for_unsupervised_learning_all_variants.py $genomic_annotation_file $variant_level_genomic_annotation_file $total_expression_outlier_file $ase_outlier_file $splicing_outlier_file $unsupervised_learning_input_dir
fi

