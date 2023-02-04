import numpy as np 
import os
import sys
import pdb







def get_tissues(tissue_names_file):
	tissues = []
	f = open(tissue_names_file)
	for line in f:
		line = line.rstrip()
		tissues.append(line)
	f.close()
	return np.asarray(tissues)

def get_used_gene_individual_pairs(input_file, num_tissues):
	gene_indi_pairs = {}
	head_count = 0
	f = open(input_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		indi_id = data[0]
		gene_id = data[1]
		pair_name = indi_id + '_' + gene_id
		if pair_name in gene_indi_pairs:
			print('assumption error!')
		gene_indi_pairs[pair_name] = np.zeros(num_tissues)
		gene_indi_pairs[pair_name].fill(np.nan)
	f.close()
	return gene_indi_pairs

def fill_in_tissues_splicing_outlier_calls(outlier_file_name, gene_individual_pairs, tissue_num):
	f = open(outlier_file_name)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			indiz = data[1:]
			continue
		ensamble_id = data[0]
		pvalues = np.asarray(data[1:]).astype(float)
		for index, pvalue in enumerate(pvalues):
			indi = indiz[index]
			pair_name = indi + '_' + ensamble_id
			if pair_name in gene_individual_pairs:
				gene_individual_pairs[pair_name][tissue_num] = pvalue
	f.close()
	return gene_individual_pairs

def convert_nan_to_NA(pval_vec):
	new_pval_vec = []
	pval_vec = np.asarray(pval_vec).astype(str)
	for ele in pval_vec:
		if ele == 'nan':
			new_pval_vec.append('NA')
		else:
			new_pval_vec.append(ele)
	return new_pval_vec

def print_tbt_to_output_file(input_file, output_file, gene_individual_pairs, tissues):
	t = open(output_file, 'w')
	f = open(input_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			num_features = len(data) - 4
			features = data[:num_features]
			t.write('\t'.join(features))
			for tissue in tissues:
				t.write('\t' + tissue + '_splicing_pvalue')
			t.write('\t' + data[-1] + '\n')
			continue
		num_features = len(data) - 4
		features = data[:num_features]
		t.write('\t'.join(features) + '\t')
		indi_id = data[0]
		gene_id = data[1]
		test_name = indi_id + '_' + gene_id
		pvalue_arr = np.asarray(gene_individual_pairs[test_name]).astype(str)
		t.write('\t'.join(pvalue_arr) + '\t' + data[-1] + '\n')
	f.close()
	t.close()

unsupervised_learning_input_dir = sys.argv[1]  # Input/outptu dir
pvalue = float(sys.argv[2]) # Threshold used for outlier calling
splicing_outlier_dir = sys.argv[3]  # Directory containing splicing outlier calls in each tissue
tissue_names_file = sys.argv[4]  # File containing list of tissue names

# Input file we will use to get list of test (individual, gene pairs to be used)
input_file = unsupervised_learning_input_dir + 'fully_observed_merged_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing_features_filter_N2_pairs.txt'

# Output file we will write results to
output_file = unsupervised_learning_input_dir + 'splicing_tbt_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing_features_filter_N2_pairs.txt'

# Extract list of tissues
tissues = get_tissues(tissue_names_file)

# Extract list of (gene,individual) pairs we are testing
gene_individual_pairs = get_used_gene_individual_pairs(input_file, len(tissues))

# Fill in gene_indiviaul_pairs with outlier pvalues
for tissue_num, tissue_name in enumerate(tissues):
	print(tissue_num)
	outlier_file_name = splicing_outlier_dir + tissue_name + '_covariate_method_none_no_global_outliers_ea_only_merged_emperical_pvalue_gene_level.txt'
	gene_individual_pairs = fill_in_tissues_splicing_outlier_calls(outlier_file_name, gene_individual_pairs, tissue_num)

# Print to output file
print_tbt_to_output_file(input_file, output_file, gene_individual_pairs, tissues)
