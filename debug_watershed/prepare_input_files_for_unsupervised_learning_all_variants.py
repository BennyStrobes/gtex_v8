import numpy as np 
import os
import sys
import pdb
import scipy.stats


def convert_na_to_nan(data_input):
	data_output = []
	for ele in data_input:
		if ele == 'NA':
			data_output.append('NaN')
		else:
			data_output.append(ele)
	return data_output

def get_splicing_and_ase_outliers(splicing_outlier_file):
	outlier_dicti = {}
	head_count = 0
	f = open(splicing_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		# Header
		if head_count == 0:
			head_count = head_count + 1
			indiz = np.asarray(data[1:])
			continue
		# Skip genes not in genes dicti
		ensamble = data[0].split('.')[0]

		# get pvalue vector
		pvalues = np.asarray(convert_na_to_nan(data[1:])).astype(float)
		# Remove columns not in individuals
		# pvalues_filtered = pvalues[good_columns]
		# Add to global pvalue_array
		for i,pvalue in enumerate(pvalues):
			indi_id = indiz[i]
			sample_name = indi_id + '_' + ensamble
			outlier_dicti[sample_name] = pvalue
	f.close()
	return outlier_dicti

def get_total_expression_outliers(total_expression_outlier_file):
	outlier_dicti = {}
	head_count = 0
	individuals = {}
	genes = {}
	f = open(total_expression_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		indi_id = data[0]
		ensamble_id = data[1].split('.')[0]
		zscore = float(data[4])
		pvalue = scipy.stats.norm.sf(abs(zscore))*2
		sample_name = indi_id + '_' + ensamble_id
		if zscore < 0:
			pvalue = pvalue*-1
		outlier_dicti[sample_name] = pvalue
		individuals[indi_id] = 1
		genes[ensamble_id] = 1
	f.close()
	for indi in individuals.keys():
		for gene in genes.keys():
			sample_name = indi + '_' + gene
			if sample_name not in outlier_dicti:
				outlier_dicti[sample_name] = float('NaN')
	return outlier_dicti

def remove_no_variance_features(input_file, fully_observed_input_file):
	f = open(fully_observed_input_file)
	col_names = {}
	head_count = 0
	for line in f:
		if head_count == 0:
			line = line.rstrip()
			data = line.split()
			if head_count == 0:
				for ele in data:
					col_names[ele] = 1
				head_count = head_count + 1
	f.close()
	output_file = input_file.split('.tx')[0] + '_features_filter.txt'
	t = open(output_file, 'w')
	head_count = 0
	f = open(input_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			good_columns = []
			for i, ele in enumerate(data):
				if ele in col_names:
					good_columns.append(i)
			good_columns = np.asarray(good_columns)
			head_count = head_count + 1

			new_data = np.asarray(data)[good_columns]
			t.write('\t'.join(new_data) + '\tN2pair\n')
			continue
		new_data = np.asarray(data)[good_columns]
		t.write('\t'.join(new_data) + '\tNA\n')
	f.close()
	t.close()

def print_outlier_output_file(splicing_outlier_dicti, te_outlier_dicti, ase_outlier_dicti, genomic_annotation_file, output_file, fully_observed_input_file):
	t = open(output_file, 'w')
	f = open(genomic_annotation_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\t' + 'splicing_pvalue\ttotal_expression_pvalue\tase_pvalue\n')
			continue
		indi_id = data[0]
		gene_id = data[1].split('.')[0]
		sample_name = indi_id + '_' + gene_id
		t.write(line)
		if sample_name in splicing_outlier_dicti:
			t.write('\t' + str(splicing_outlier_dicti[sample_name]))
		else:
			t.write('\tnan')
		if sample_name in te_outlier_dicti:
			t.write('\t' + str(te_outlier_dicti[sample_name]))
		else:
			t.write('\tnan')
		if sample_name in ase_outlier_dicti:
			t.write('\t' + str(ase_outlier_dicti[sample_name]))
		else:
			t.write('\tnan')
		t.write('\n')
	f.close()
	t.close()
	remove_no_variance_features(output_file, fully_observed_input_file)
	#add_n2_pairs_column(output_file.split('.tx')[0] + '_features_filter.txt',gene_individual_to_variant_mapping_file)
	#os.system('rm ' + output_file)
	#os.system('rm ' + output_file.split('.tx')[0] + '_features_filter.txt')

def remove_fully_missing(input_file, output_file):
	head_count = 0
	f = open(input_file)
	t = open(output_file, 'w')
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\n')
			continue
		if data[-2] == 'nan' and data[-3] == 'nan' and data[-4] == 'nan':
			continue
		t.write(line + '\n')
	f.close()
	t.close()

genomic_annotation_file = sys.argv[1]
variant_level_genomic_annotation_file = sys.argv[2]
total_expression_outlier_file = sys.argv[3]
ase_outlier_file = sys.argv[4]
splicing_outlier_file = sys.argv[5]
unsupervised_learning_input_dir = sys.argv[6]

print("START")

fully_observed_input_file = unsupervised_learning_input_dir + 'fully_observed_merged_outliers_0.01_genes_intersection_between_te_ase_splicing_features_filter_N2_pairs.txt'

splicing_outliers = get_splicing_and_ase_outliers(splicing_outlier_file)
ase_outliers = get_splicing_and_ase_outliers(ase_outlier_file)
total_expression_outliers = get_total_expression_outliers(total_expression_outlier_file)

print_outlier_output_file(splicing_outliers, total_expression_outliers, ase_outliers, variant_level_genomic_annotation_file, unsupervised_learning_input_dir + 'all_availibile_samples_variant_level.txt', fully_observed_input_file)
remove_fully_missing(unsupervised_learning_input_dir + 'all_availibile_samples_variant_level_features_filter.txt', unsupervised_learning_input_dir + 'all_availibile_samples_variant_level_features_filter_partially_observed_expression.txt')


print_outlier_output_file(splicing_outliers, total_expression_outliers, ase_outliers, genomic_annotation_file, unsupervised_learning_input_dir + 'all_availibile_samples.txt', fully_observed_input_file)
remove_fully_missing(unsupervised_learning_input_dir + 'all_availibile_samples_features_filter.txt', unsupervised_learning_input_dir + 'all_availibile_samples_features_filter_partially_observed_expression.txt')



