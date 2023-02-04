import numpy as np 
import os
import sys
import pdb
import scipy.stats


def get_total_expression_individuals(total_expression_outlier_file):
	indi = {}
	head_count = 0
	f = open(total_expression_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		indi[data[0]] = 1
	f.close()
	return indi


def get_splicing_and_ase_individuals(splicing_outlier_file):
	indi = {}
	head_count = 0
	f = open(splicing_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			for ele in data[1:]:
				indi[ele] = 1
	f.close()
	return indi

# Extract dictionary list of individuals used in all three methods
def get_list_of_individuals_used_in_all_methods(total_expression_outlier_file, ase_outlier_file, splicing_outlier_file):
	total_expression_individuals = get_total_expression_individuals(total_expression_outlier_file)
	ase_individuals = get_splicing_and_ase_individuals(ase_outlier_file)
	splicing_individuals = get_splicing_and_ase_individuals(splicing_outlier_file)
	# Take union of the above three dictionaries
	union_individuals = {}
	for indi in splicing_individuals.keys():
		if indi in ase_individuals and indi in total_expression_individuals:
			union_individuals[indi] = 1
	return union_individuals

def get_total_expression_genes(total_expression_outlier_file):
	genes = {}
	head_count = 0
	f = open(total_expression_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		genes[data[1].split('.')[0]] = 1
	f.close()
	return genes

def get_ase_genes(ase_outlier_file):
	genes = {}
	f = open(ase_outlier_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		genes[data[0]] = 1
	f.close()
	return genes

def get_splicing_genes(splicing_outlier_file):
	f = open(splicing_outlier_file)
	head_count = 0
	genes = {}
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		ensamble = data[0].split('.')[0]
		genes[ensamble] = 1
	f.close()
	return genes

# Extract dictionary list of genes used in all three methods
def get_list_of_genes_used_in_all_methods(total_expression_outlier_file, ase_outlier_file, splicing_outlier_file):
	splicing_genes = get_splicing_genes(splicing_outlier_file)
	ase_genes = get_ase_genes(ase_outlier_file)
	total_expression_genes = get_total_expression_genes(total_expression_outlier_file)
	union_genes = {}
	for gene in splicing_genes.keys():
		if gene in ase_genes and gene in total_expression_genes:
			union_genes[gene] = 1
	return union_genes

def convert_na_to_nan(data_input):
	data_output = []
	for ele in data_input:
		if ele == 'NA':
			data_output.append('NaN')
		else:
			data_output.append(ele)
	return data_output

# For splicing outliers extract list of genes for which we have at least one outlier sample and extract pvalue threshold that gives us num_outlier_samples
def get_splicing_outlier_info(individuals, genes, pvalue_threshold, splicing_outlier_file):
	splicing_outlier_genes = {}
	head_count = 0
	f = open(splicing_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		# Header
		if head_count == 0:
			head_count = head_count + 1
			indiz = np.asarray(data[1:])
			# Get array of columns that map to individuals we are using
			good_columns = []
			for i,indi in enumerate(indiz):
				if indi in individuals:
					good_columns.append(i)
			# ERROR CHECKING
			new_indiz = indiz[good_columns]
			for indi in new_indiz:
				if indi not in individuals:
					print('assumption error')
					pdb.set_trace()
			continue
		# Skip genes not in genes dicti
		ensamble = data[0].split('.')[0]
		if ensamble not in genes:
			continue
		# get pvalue vector
		pvalues = np.asarray(convert_na_to_nan(data[1:])).astype(float)
		# Remove columns not in individuals
		pvalues_filtered = pvalues[good_columns]
		for pvalue in pvalues_filtered:
			if np.isnan(pvalue):
				continue
			# Check if (individual, gene) is an outlier. If so, add gene to dictionary
			if pvalue < pvalue_threshold:
				splicing_outlier_genes[ensamble] = 1
	f.close()
	return splicing_outlier_genes


# For total expression outliers extract list of genes for which we have at least one outlier sample and extract pvalue threshold that gives us num_outlier_samples
def get_total_expression_outlier_info(individuals, genes, pvalue_threshold, total_expression_outlier_file):
	total_expression_outlier_genes = {}
	f = open(total_expression_outlier_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		indi_id = data[0]
		ensamble_id = data[1].split('.')[0]
		if indi_id not in individuals:
			continue
		if ensamble_id not in genes:
			continue
		zscore = abs(float(data[4]))
		pvalue = scipy.stats.norm.sf(zscore)*2
		if pvalue < pvalue_threshold:
			total_expression_outlier_genes[ensamble_id] = 1
	f.close()
	return total_expression_outlier_genes

def get_ase_outlier_info(individuals, genes, num_outlier_samples, ase_outlier_file):
	pvalue_array = []
	f = open(ase_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		indi_id = data[0]
		ensamble_id = data[1]
		if indi_id not in individuals:
			continue
		if ensamble_id not in genes:
			continue
		pvalue_array.append(float(data[2]))
	f.close()
	# Sort pvalue array
	sorted_pvalue_array = sorted(pvalue_array)
	# Get pvalue threshold
	if sorted_pvalue_array[num_outlier_samples-1] == sorted_pvalue_array[num_outlier_samples]:
		print('assumption error')
		pdb.set_trace()
	pvalue_threshold = (sorted_pvalue_array[num_outlier_samples-1] + sorted_pvalue_array[num_outlier_samples])/2.0
	ase_outlier_genes = {}
	f = open(ase_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		indi_id = data[0]
		ensamble_id = data[1]
		if indi_id not in individuals:
			continue
		if ensamble_id not in genes:
			continue
		pvalue = float(data[2])
		if pvalue < pvalue_threshold:
			ase_outlier_genes[ensamble_id] = 1
	f.close()
	return ase_outlier_genes, pvalue_threshold

def union_of_three_dictionaries(dicti1, dicti2):
	union = {}
	for key  in dicti1.keys():
		union[key] = 1
	for key  in dicti2.keys():
		union[key] = 1	
	return union


def intersection_of_three_dictionaries(dicti1, dicti2, dicti3):
	intersection = {}
	for key  in dicti1.keys():
		if key in dicti2 and key in dicti3:
			intersection[key] = 1
	return intersection


def get_splicing_outliers(genes, individuals, splicing_outlier_file):
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
			# Get array of columns that map to individuals we are using
			good_columns = []
			for i,indi in enumerate(indiz):
				if indi in individuals:
					good_columns.append(i)
			# ERROR CHECKING
			new_indiz = indiz[good_columns]
			for indi in new_indiz:
				if indi not in individuals:
					print('assumption error')
					pdb.set_trace()
			continue
		# Skip genes not in genes dicti
		ensamble = data[0].split('.')[0]
		if ensamble not in genes:
			continue
		# get pvalue vector
		pvalues = np.asarray(convert_na_to_nan(data[1:])).astype(float)
		# Remove columns not in individuals
		pvalues_filtered = pvalues[good_columns]
		# Add to global pvalue_array
		for i,pvalue in enumerate(pvalues_filtered):
			indi_id = new_indiz[i]
			sample_name = indi_id + '_' + ensamble
			outlier_dicti[sample_name] = pvalue
	f.close()
	return outlier_dicti

def get_total_expression_outliers(genes, individuals, total_expression_outlier_file):
	outlier_dicti = {}
	head_count = 0
	f = open(total_expression_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		indi_id = data[0]
		ensamble_id = data[1].split('.')[0]
		if indi_id not in individuals:
			continue
		if ensamble_id not in genes:
			continue
		zscore = float(data[4])
		pvalue = scipy.stats.norm.sf(abs(zscore))*2
		sample_name = indi_id + '_' + ensamble_id
		if zscore < 0:
			pvalue = pvalue*-1
		outlier_dicti[sample_name] = pvalue
	f.close()
	for indi in individuals.keys():
		for gene in genes.keys():
			sample_name = indi + '_' + gene
			if sample_name not in outlier_dicti:
				outlier_dicti[sample_name] = float('NaN')
	return outlier_dicti

def get_ase_outliers(genes, individuals, ase_pvalue_threshold, ase_outlier_file):
	outlier_dicti = {}
	f = open(ase_outlier_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		indi_id = data[0]
		ensamble_id = data[1]
		if indi_id not in individuals:
			continue
		if ensamble_id not in genes:
			continue
		pvalue = float(data[2])
		sample_name = indi_id + '_' + ensamble_id
		if pvalue <  ase_pvalue_threshold:
			outlier_dicti[sample_name] = 1
	f.close()
	for indi in individuals.keys():
		for gene in genes.keys():
			sample_name = indi + '_' + gene
			if sample_name not in outlier_dicti:
				outlier_dicti[sample_name] = 0
	return outlier_dicti


def print_outlier_output_file(outlier_dicti, genomic_annotation_file, output_file, gene_individual_to_variant_mapping_file):
	t = open(output_file, 'w')
	f = open(genomic_annotation_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\t' + 'pvalue\n')
			continue
		indi_id = data[0]
		gene_id = data[1].split('.')[0]
		sample_name = indi_id + '_' + gene_id
		if sample_name in outlier_dicti:
			t.write(line + '\t' + str(outlier_dicti[sample_name]) + '\n')
	f.close()
	t.close()
	remove_no_variance_features(output_file)
	add_n2_pairs_column(output_file.split('.tx')[0] + '_features_filter.txt',gene_individual_to_variant_mapping_file)
	os.system('rm ' + output_file)
	os.system('rm ' + output_file.split('.tx')[0] + '_features_filter.txt')


def print_outlier_output_file_no_nan(outlier_dicti1, outlier_dicti2, outlier_dicti3, genomic_annotation_file, output_file, gene_individual_to_variant_mapping_file):
	t = open(output_file, 'w')
	f = open(genomic_annotation_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\t' + 'pvalue\n')
			continue
		indi_id = data[0]
		gene_id = data[1].split('.')[0]
		sample_name = indi_id + '_' + gene_id
		if sample_name in outlier_dicti1 and sample_name in outlier_dicti2 and sample_name in outlier_dicti3:
			if np.isnan(outlier_dicti1[sample_name]) == False and np.isnan(outlier_dicti2[sample_name]) == False and np.isnan(outlier_dicti3[sample_name]) == False:
				t.write(line + '\t' + str(outlier_dicti1[sample_name]) + '\n')
	f.close()
	t.close()
	remove_no_variance_features(output_file)
	add_n2_pairs_column(output_file.split('.tx')[0] + '_features_filter.txt',gene_individual_to_variant_mapping_file)
	os.system('rm ' + output_file)
	os.system('rm ' + output_file.split('.tx')[0] + '_features_filter.txt')


def add_n2_pairs_column(input_file, gene_individual_to_variant_mapping_file):
	f = open(gene_individual_to_variant_mapping_file)
	test_to_variant_mapping = {}
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		test_name = data[0] + '_' + data[1]
		variants = data[1] + '_' + ','.join(sorted(data[2].split(',')))
		if test_name in test_to_variant_mapping:
			print('assumption error')
			pdb.set_trace()
		test_to_variant_mapping[test_name] = variants
	f.close()
	dicti = {}
	output_file = input_file.split('.tx')[0] + '_N2_pairs.txt'
	f = open(input_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			header = line
			continue
		test_name = data[0] + '_' + data[1]
		variants_mapped = test_to_variant_mapping[test_name]
		liner_long = '\t'.join(data)
		if variants_mapped not in dicti:
			dicti[variants_mapped] = []
		dicti[variants_mapped].append(liner_long)
	f.close()
	t = open(output_file,'w')
	t.write(header + '\tN2pair\n')
	for key in sorted(dicti.keys()):
		arr = dicti[key]
		if len(arr) == 1:
			liner_long = arr[0]
			t.write(liner_long + '\t' + 'NA' + '\n')
	n2_pair_counter = 0
	for key in sorted(dicti.keys()):
		arr = dicti[key]
		if len(arr) > 1:
			n2_pair_counter = n2_pair_counter + 1
			for liner_long in arr[:2]:
				t.write(liner_long + '\t' + str(n2_pair_counter) + '\n')
	t.close()

def remove_no_variance_features(input_file):
	output_file = input_file.split('.tx')[0] + '_features_filter.txt'
	f = open(input_file)
	arr = []
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			for i, ele in enumerate(data):
				arr.append({})
			continue
		for i,ele in enumerate(data):
			arr[i][ele] = 1
	f.close()
	good_columns = []
	for i, dicti in enumerate(arr):
		if len(dicti) > 1:
			good_columns.append(i)
	t = open(output_file, 'w')
	head_count = 0
	f = open(input_file)
	for line in f:
		line = line.rstrip()
		data = line.split()
		new_data = np.asarray(data)[good_columns]
		t.write('\t'.join(new_data) + '\n')
	f.close()
	t.close()

def merge_three_files(input_file1, input_file2, input_file3, output_file):
	# Loop through input file1
	# Keep track of feature vectors as well as pvalue from outlier status
	file_1_feature_vectors = []
	file_1_pvalues = []
	f = open(input_file1)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		pvalue = data[-2]
		feature_vector = '\t'.join(data[0:-2])
		file_1_pvalues.append(pvalue)
		file_1_feature_vectors.append(feature_vector)
	f.close()
	file_2_feature_vectors = []
	file_2_pvalues = []
	f = open(input_file2)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			continue
		pvalue = data[-2]
		feature_vector = '\t'.join(data[0:-2])
		file_2_pvalues.append(pvalue)
		file_2_feature_vectors.append(feature_vector)
	f.close()


	t = open(output_file,'w')
	f = open(input_file3)
	head_count = 0
	counter = -1
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			t.write('\t'.join(data[0:-2]) + '\tsplicing_pvalue\ttotal_expression_pvalue\tase_pvalue\t' + data[-1] + '\n')
			continue
		pvalue = data[-2]
		feature_vector = '\t'.join(data[0:-2])
		counter = counter + 1
		if feature_vector != file_1_feature_vectors[counter] or feature_vector != file_2_feature_vectors[counter]:
			print('assumption errror')
			pdb.set_trace()
		t.write(feature_vector + '\t' + file_1_pvalues[counter] + '\t' + file_2_pvalues[counter] + '\t' + pvalue + '\t' + data[-1] + '\n')
	f.close()
	t.close()



genomic_annotation_file = sys.argv[1]
total_expression_outlier_file = sys.argv[2]
ase_outlier_file = sys.argv[3]
splicing_outlier_file = sys.argv[4]
unsupervised_learning_input_dir = sys.argv[5]
pvalue = float(sys.argv[6])
gene_individual_to_variant_mapping_file = sys.argv[7]

# Extract dictionary list of individuals used in all three methods
individuals = get_list_of_individuals_used_in_all_methods(total_expression_outlier_file, ase_outlier_file, splicing_outlier_file)
# Extract dictionary list of genes used in all three methods
genes = get_list_of_genes_used_in_all_methods(total_expression_outlier_file, ase_outlier_file, splicing_outlier_file)


# For ase outliers extract list of genes for which we have at least one outlier sample and extract pvalue threshold that gives us num_outlier_samples
ase_outlier_genes = get_splicing_outlier_info(individuals, genes, pvalue, ase_outlier_file)
# For splicing outliers extract list of genes for which we have at least one outlier sample and extract pvalue threshold that gives us num_outlier_samples
splicing_outlier_genes = get_splicing_outlier_info(individuals, genes, pvalue, splicing_outlier_file)
# For total expression outliers extract list of genes for which we have at least one outlier sample and extract zscore threshold that gives us num_outlier_samples
total_expression_outlier_genes = get_total_expression_outlier_info(individuals, genes, pvalue, total_expression_outlier_file)

# Take intersection of outlier genes from each of three methods
outlier_genes = intersection_of_three_dictionaries(splicing_outlier_genes, total_expression_outlier_genes, ase_outlier_genes)


# Extract dictionary of (sample, gene) pairs for each class that are valid. Value of dictionary is 1 if outlier and 0 if inlier
splicing_outliers = get_splicing_outliers(outlier_genes, individuals, splicing_outlier_file)
total_expression_outliers = get_total_expression_outliers(outlier_genes, individuals, total_expression_outlier_file)
ase_outliers = get_splicing_outliers(outlier_genes, individuals, ase_outlier_file)



# Print outliers
ase_output_file = unsupervised_learning_input_dir + 'ase_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing.txt'
print_outlier_output_file(ase_outliers, genomic_annotation_file, ase_output_file, gene_individual_to_variant_mapping_file)
splicing_output_file = unsupervised_learning_input_dir + 'splicing_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing.txt'
print_outlier_output_file(splicing_outliers, genomic_annotation_file, splicing_output_file, gene_individual_to_variant_mapping_file)
total_expression_output_file = unsupervised_learning_input_dir + 'total_expression_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing.txt'
print_outlier_output_file(total_expression_outliers, genomic_annotation_file, total_expression_output_file, gene_individual_to_variant_mapping_file)

# Merge three outlier files
merged_output_file = unsupervised_learning_input_dir + 'merged_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing_features_filter_N2_pairs.txt'
merge_three_files(splicing_output_file.split('.tx')[0] + '_features_filter_N2_pairs.txt', total_expression_output_file.split('.tx')[0] + '_features_filter_N2_pairs.txt', ase_output_file.split('.tx')[0] + '_features_filter_N2_pairs.txt', merged_output_file)







# Print outliers
splicing_output_file = unsupervised_learning_input_dir + 'fully_observed_splicing_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing.txt'
print_outlier_output_file_no_nan(splicing_outliers, total_expression_outliers,ase_outliers, genomic_annotation_file, splicing_output_file, gene_individual_to_variant_mapping_file)
total_expression_output_file = unsupervised_learning_input_dir + 'fully_observed_total_expression_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing.txt'
print_outlier_output_file_no_nan(total_expression_outliers, splicing_outliers, ase_outliers, genomic_annotation_file, total_expression_output_file, gene_individual_to_variant_mapping_file)
ase_output_file = unsupervised_learning_input_dir + 'fully_observed_ase_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing.txt'
print_outlier_output_file_no_nan(ase_outliers, total_expression_outliers, splicing_outliers, genomic_annotation_file, ase_output_file, gene_individual_to_variant_mapping_file)



# Merge three outlier files
merged_output_file = unsupervised_learning_input_dir + 'fully_observed_merged_outliers_' + str(pvalue) + '_genes_intersection_between_te_ase_splicing_features_filter_N2_pairs.txt'
merge_three_files(splicing_output_file.split('.tx')[0] + '_features_filter_N2_pairs.txt', total_expression_output_file.split('.tx')[0] + '_features_filter_N2_pairs.txt', ase_output_file.split('.tx')[0] + '_features_filter_N2_pairs.txt', merged_output_file)





