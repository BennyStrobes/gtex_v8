args = commandArgs(trailingOnly=TRUE)
library(glmnet)
library(methods)
library(stats)
library(utils)
library(Biobase)
library(pROC)
library(ggplot2)
library(sigmoid)
library(Rcpp)
library(numDeriv)
library(lbfgs)
library(reshape)
library(grid)
library(PRROC)
library(cowplot)
library(RColorBrewer)
source("watershed.R")





make_posterior_predictions_object_exact_inference <- function(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier_component, phi_outlier_component, number_of_dimensions) {
	prediction_output <- compute_all_exact_posterior_predictions_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier_component, phi_outlier_component, number_of_dimensions)

	predictions_list <- list()
	for (num_sample in 1:nrow(feat)) {
		sample_list <- list()
		for (combination_number in 1:nrow(prediction_output$combination)) {
			sample_list[[paste(prediction_output$combination[combination_number,]+1,collapse=" ")]] = prediction_output$probability[num_sample, combination_number]
		}
		predictions_list[[num_sample]] = sample_list
	}
	return(predictions_list)

}

make_posterior_predictions_object_vi <- function(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier_component, phi_outlier_component, number_of_dimensions, posterior_prob_test) {
 	prediction_output <- compute_all_exact_posterior_predictions_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier_component, phi_outlier_component, number_of_dimensions)

	predictions_list <- list()
	for (num_sample in 1:nrow(feat)) {
		sample_list <- list()
		for (combination_number in 1:nrow(prediction_output$combination)) {
			prob <- 1
			for (dimension in 1:number_of_dimensions) {
				if (prediction_output$combination[combination_number,dimension] == 1) {
					prob <- prob*posterior_prob_test[num_sample, dimension]
				} else {
					prob <- prob*(1.0-posterior_prob_test[num_sample, dimension])
				}
			}
			sample_list[[paste(prediction_output$combination[combination_number,]+1,collapse=" ")]] = prob

		}
		predictions_list[[num_sample]] = sample_list
	}
	return(predictions_list)

}

compute_output_probabilities <- function(discrete_outliers_test2, predictions_object) {
	pvalues <- c()
	for (sample_num in 1:nrow(discrete_outliers_test2)) {
		if (sum(discrete_outliers_test2[sample_num,] == numeric(ncol(discrete_outliers_test2)) + 1) != ncol(discrete_outliers_test2)) {
			prob <- predictions_object[[sample_num]][[paste(discrete_outliers_test2[sample_num,], collapse=" ")]]
			pvalues <- c(pvalues,prob)
		}
	}
	return(pvalues)
}

compute_accuracy <- function(discrete_outliers_test2, predictions_object) {
	correct_count = 0
	total_count = 0
	for (sample_num in 1:nrow(discrete_outliers_test2)) {
		gold_standard <- paste(discrete_outliers_test2[sample_num,], collapse=" ")
		max_value = -1
		for (combo_num in 1:length(predictions_object[[sample_num]])) {
			val = predictions_object[[sample_num]][[combo_num]]
			if (val > max_value) {
				max_value = val
				name = labels(predictions_object[[sample_num]][combo_num])
			}
			if (name == gold_standard) {
				correct_count = correct_count + 1
			}
			total_count = total_count + 1
		}
	}
	return(correct_count/total_count)
}


# Confusion matrix where rows are actual labels and columns are predicted labels
# Each row (actual label) is normalized by the sum across all columns (predicted label) in that row (actual label)
make_confusion_matrix <- function(predictions_object, binary_outliers_test2) {
	# Initialize confusion matrix to enries of zero and row names and row labels
	num_classes <- length(labels(predictions_object[[1]]))
	confusion_matrix <- matrix(0,num_classes, num_classes)
	rownames(confusion_matrix) = labels(predictions_object[[1]])
	colnames(confusion_matrix) = labels(predictions_object[[1]])
	# Loop through each test sample
	for (sample_num in 1:nrow(binary_outliers_test2)) {
		# Pseudo-gold standard label
		gold_standard <- paste(binary_outliers_test2[sample_num,], collapse=" ")
		gold_index <- which(labels(predictions_object[[sample_num]]) == gold_standard)
		# Get predicted label according to maximum posterior probability
		max_value = -1
		max_index = -1
		for (combo_num in 1:length(predictions_object[[sample_num]])) {
			val = predictions_object[[sample_num]][[combo_num]]
			if (val > max_value) {
				max_value = val
				max_index <- combo_num
			}
		}
		# Add count to confusion matrix
		confusion_matrix[gold_index, max_index] = confusion_matrix[gold_index,max_index] + 1
	}
	#confusion_matrix = confusion_matrix/rowSums(confusion_matrix)
	return(confusion_matrix)
}



# Confusion matrix where rows are actual labels and columns are predicted labels
generate_confusion_matrix <- function(feat_test, discrete_outliers_test1, binary_outliers_test2, watershed_model, inference_method, posterior_prob_test) {
	if (inference_method == "exact") {
		predictions_object <- make_posterior_predictions_object_exact_inference(feat_test, discrete_outliers_test1, watershed_model$theta_singleton, watershed_model$theta_pair, watershed_model$theta, watershed_model$phi$inlier_component, watershed_model$phi$outlier_component, watershed_model$number_of_dimensions)
	} else if (inference_method == "vi") {
		predictions_object <- make_posterior_predictions_object_vi(feat_test, discrete_outliers_test1, watershed_model$theta_singleton, watershed_model$theta_pair, watershed_model$theta, watershed_model$phi$inlier_component, watershed_model$phi$outlier_component, watershed_model$number_of_dimensions, posterior_prob_test)
	}
	confusion_matrix <- make_confusion_matrix(predictions_object, binary_outliers_test2+1)  	
	return(confusion_matrix)
}

# Helper function to remove NAs from vector
remove_na <- function(x) {
	return(x[!is.na(x)])
}

 #######################################
# Extract ROC curves and precision recall curves for test set (in each dimension seperately) using:
#### 1. Watershed
#### 2. GAM
#### 3. RNA-only
#######################################
compute_roc_across_dimensions <- function(number_of_dimensions, dimension_labels, posterior_prob_test, real_valued_outliers_test1, gam_posteriors, binary_outliers_test2) {
	roc_object_across_dimensions <- list()
	pos_list <- c()
	neg_list <- c()
	med_pos_list <- c()
	med_neg_list <- c()

	median_predictions <- readRDS("/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/unsupervised_modeling/watershed_tbt_roc/non_tissue_specific_predictions.rds")
	median_dimension <- 1

  	# Loop through dimensions
  	for (dimension in 1:number_of_dimensions) {
  		# Name of dimension
  		dimension_name <- strsplit(dimension_labels[dimension],"_pval")[[1]][1]
  		# Pseudo gold standard
  		test_outlier_status <- binary_outliers_test2[,dimension]
  		# river predictions
  		# roc_obj <- roc(test_outlier_status, posterior_prob_test[,dimension])
  		roc_obj <- roc.curve(scores.class0 = remove_na(posterior_prob_test[,dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]), scores.class1 = remove_na(posterior_prob_test[,dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]), curve = T)
  		pr_obj <- pr.curve(scores.class0 = remove_na(posterior_prob_test[,dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]), scores.class1 = remove_na(posterior_prob_test[,dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]), curve = T)
  		
  		pos_list <- c(pos_list, remove_na(posterior_prob_test[,dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]))
  		neg_list <- c(neg_list, remove_na(posterior_prob_test[,dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]))

  		med_pos_list <- c(med_pos_list, remove_na(median_predictions[,median_dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]))
  		med_neg_list <- c(med_neg_list, remove_na(median_predictions[,median_dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]))
  		# Predictions with only RNA
  		#rna_only_roc_obj <- roc(test_outlier_status, real_valued_outliers_test1[,dimension])
  		rna_only_roc_obj <- roc.curve(scores.class0 = remove_na(real_valued_outliers_test1[,dimension][test_outlier_status==1]), scores.class1 = remove_na(real_valued_outliers_test1[,dimension][test_outlier_status==0]), curve = T)
  		rna_only_pr_obj <- pr.curve(scores.class0 = remove_na(real_valued_outliers_test1[,dimension][test_outlier_status==1]), scores.class1 = remove_na(real_valued_outliers_test1[,dimension][test_outlier_status==0]), curve = T)

  		# predictions with only genomic annotations
  		#gam_roc_obj <- roc(test_outlier_status, gam_posteriors[,dimension])
   		gam_roc_obj <- roc.curve(scores.class0 = remove_na(gam_posteriors[,dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]), scores.class1 = remove_na(gam_posteriors[,dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]), curve = T)
   		gam_pr_obj <- pr.curve(scores.class0 = remove_na(gam_posteriors[,dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]), scores.class1 = remove_na(gam_posteriors[,dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]), curve = T)

   		median_roc_obj <- roc.curve(scores.class0 = remove_na(median_predictions[,median_dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]), scores.class1 = remove_na(median_predictions[,median_dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]), curve = T)
   		median_pr_obj <- pr.curve(scores.class0 = remove_na(median_predictions[,median_dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])]), scores.class1 = remove_na(median_predictions[,median_dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])]), curve = T)

		evaROC <-	
		 list(watershed_sens=roc_obj$curve[,2],
              watershed_spec=1-roc_obj$curve[,1],
         	  watershed_auc=roc_obj$auc,
         	  watershed_pr_auc=pr_obj$auc.integral,
         	  watershed_recall=pr_obj$curve[,1],
         	  watershed_precision=pr_obj$curve[,2],
         	  GAM_sens=gam_roc_obj$curve[,2],
              GAM_spec=1-gam_roc_obj$curve[,1],
              GAM_auc=gam_roc_obj$auc,
         	  GAM_pr_auc=gam_pr_obj$auc.integral,
         	  GAM_recall=gam_pr_obj$curve[,1],
         	  GAM_precision=gam_pr_obj$curve[,2],
         	  rna_only_pr_auc=rna_only_pr_obj$auc.integral,
         	  rna_only_recall=rna_only_pr_obj$curve[,1],
         	  rna_only_precision=rna_only_pr_obj$curve[,2],
              rna_only_sens=rna_only_roc_obj$curve[,2],
              rna_only_spec=1-rna_only_roc_obj$curve[,1],
              rna_only_auc=rna_only_roc_obj$auc,
              median_pr_auc=median_pr_obj$auc.integral,
              median_recall=median_pr_obj$curve[,1],
              median_precision=median_pr_obj$curve[,2],
              num_positive_pairs=length(remove_na(posterior_prob_test[,dimension][test_outlier_status==1 & !is.na(real_valued_outliers_test1[,dimension])])),
              num_negative_pairs=length(remove_na(posterior_prob_test[,dimension][test_outlier_status==0 & !is.na(real_valued_outliers_test1[,dimension])])))


		 roc_object_across_dimensions[[dimension]] <- list(name=dimension_name, evaROC=evaROC)
	}

		pr_obj <- pr.curve(scores.class0=pos_list, scores.class1=neg_list, curve = T)

		med_pr_obj <- pr.curve(scores.class0=med_pos_list, scores.class1=med_neg_list, curve=T)


		print(length(pos_list))
		print(length(med_pos_list))
		evaROC <-	
		 list(watershed_pr_auc=pr_obj$auc.integral,
         	  watershed_recall=pr_obj$curve[,1],
         	  watershed_precision=pr_obj$curve[,2],
         	  median_pr_auc=med_pr_obj$auc.integral,
         	  median_recall=med_pr_obj$curve[,1],
         	  median_precision=med_pr_obj$curve[,2])

		roc_object_across_dimensions[[(number_of_dimensions + 1)]] <- list(name="joint", evaROC=evaROC)

	return(roc_object_across_dimensions)
}























