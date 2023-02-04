args = commandArgs(trailingOnly=TRUE)
source("watershed.R")
library(cowplot)
library(RColorBrewer)
library(ggplot2)
library(Biobase)

initialize_phi<- function(num_bins,dim) {
  phi_outlier <- matrix(1,dim,num_bins)
  phi_inlier <- matrix(1,dim,num_bins)
  phi_inlier[,1] = .8
  phi_inlier[,2] = .1
  phi_inlier[,3] = .1
 

  phi_outlier[,1] = .01
  phi_outlier[,2] = .29
  phi_outlier[,3] = .7


  ####################
  # Total expression
  ####################
  phi_inlier[2,1] = .05
  phi_inlier[2,2] = .9
  phi_inlier[2,3] = .05


  phi_outlier[2,1] = .49
  phi_outlier[2,2] = .02
  phi_outlier[2,3] = .49



  phi_init <- list(inlier_component = phi_inlier, outlier_component = phi_outlier)
  return(phi_init)
}

 #######################################
# Extract ROC curves and precision recall curves for test set (in each dimension seperately) using:
#### 1. Watershed
#### 2. GAM
#### 3. RNA-only
#######################################
compute_roc_across_dimensions <- function(number_of_dimensions, dimension_labels, posterior_prob_test, real_valued_outliers_test1, gam_posteriors, binary_outliers_test2) {
  roc_object_across_dimensions <- list()
    # Loop through dimensions
    for (dimension in 1:number_of_dimensions) {
      # Name of dimension
      dimension_name <- strsplit(dimension_labels[dimension],"_pval")[[1]][1]
      # Pseudo gold standard
      test_outlier_status <- binary_outliers_test2[,dimension]
      # river predictions
      # roc_obj <- roc(test_outlier_status, posterior_prob_test[,dimension])
      roc_obj <- roc.curve(scores.class0 = posterior_prob_test[,dimension][test_outlier_status==1], scores.class1 = posterior_prob_test[,dimension][test_outlier_status==0], curve = T)
      pr_obj <- pr.curve(scores.class0 = posterior_prob_test[,dimension][test_outlier_status==1], scores.class1 = posterior_prob_test[,dimension][test_outlier_status==0], curve = T)
      # Predictions with only RNA
      #rna_only_roc_obj <- roc(test_outlier_status, real_valued_outliers_test1[,dimension])
      rna_only_roc_obj <- roc.curve(scores.class0 = real_valued_outliers_test1[,dimension][test_outlier_status==1], scores.class1 = real_valued_outliers_test1[,dimension][test_outlier_status==0], curve = T)
      rna_only_pr_obj <- pr.curve(scores.class0 = real_valued_outliers_test1[,dimension][test_outlier_status==1], scores.class1 = real_valued_outliers_test1[,dimension][test_outlier_status==0], curve = T)

      # predictions with only genomic annotations
      #gam_roc_obj <- roc(test_outlier_status, gam_posteriors[,dimension])
      gam_roc_obj <- roc.curve(scores.class0 = gam_posteriors[,dimension][test_outlier_status==1], scores.class1 = gam_posteriors[,dimension][test_outlier_status==0], curve = T)
      gam_pr_obj <- pr.curve(scores.class0 = gam_posteriors[,dimension][test_outlier_status==1], scores.class1 = gam_posteriors[,dimension][test_outlier_status==0], curve = T)


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
              rna_only_auc=rna_only_roc_obj$auc)


     roc_object_across_dimensions[[dimension]] <- list(name=dimension_name, evaROC=evaROC)
  }
  return(roc_object_across_dimensions)
}

extract_pairwise_observed_labels <- function(binary_outliers_train) {
  nrow <- dim(binary_outliers_train)[1]
  num_dim <- dim(binary_outliers_train)[2]

  pairwise_mat <- matrix(0, nrow, choose(number_of_dimensions, 2))

  dimension_counter <- 1
  for (dim1 in 1:num_dim) {
    for (dim2 in dim1:num_dim) {
      if (dim1 != dim2) {
        pairwise_mat[,dimension_counter] <- binary_outliers_train[,dim1]*binary_outliers_train[,dim2]
        dimension_counter <- dimension_counter + 1
      }
    }
  }
  return(pairwise_mat)
}

initialize_genomic_annotation_variables <- function(number_of_features, number_of_dimensions, independent_variables) {
  if (independent_variables == "true") {
    pair_value = 0
    theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
  } else if (independent_variables == "false") {
    pair_value = 1e-7
    theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
  } else if (independent_variables == "false_geno") {
    pair_value = 1e-7
    theta_pair = matrix(0, number_of_features+1, choose(number_of_dimensions, 2))
    theta_pair[1,] <- numeric(choose(number_of_dimensions, 2)) + pair_value
  }

  beta_init = matrix(0,number_of_features+1, number_of_dimensions)
  theta_singleton = beta_init[1,]
  theta = beta_init[2:(number_of_features + 1),]
  # Initialize vector
  x <- c()
  # Add theta_singleton (intercepts)
  x <- c(x, theta_singleton)
  # Add theta for each dimension (betas)
  for (dimension in 1:number_of_dimensions) {
    x <- c(x, theta[, dimension])
  }
  # Add theta_pair (edges between unobserved nodes)
  for (row_number in 1:(dim(theta_pair)[1])) {
    x <- c(x, theta_pair[row_number,])
  }
  return(x)
}

genomic_annotation_model_cv <- function(feat_train, binary_outliers_train, nfolds, costs, independent_variables) {
  number_of_dimensions <- dim(binary_outliers_train)[2]
  number_of_features <- dim(feat_train)[2]

  gradient_variable_vec <- initialize_genomic_annotation_variables(number_of_features, number_of_dimensions, independent_variables)

  if (independent_variables == "true") {
    pair_value = 0
    theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
  } else if (independent_variables == "false") {
    pair_value = 1e-7
    theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
  } else if (independent_variables == "false_geno") {
    pair_value = 1e-7
    theta_pair = matrix(0, number_of_features+1, choose(number_of_dimensions, 2))
    theta_pair[1,] <- numeric(choose(number_of_dimensions, 2)) + pair_value
  }

  # Create GAM parameters section
  beta_init = matrix(0,number_of_features+1, number_of_dimensions)
  theta_singleton = beta_init[1,]
  theta = beta_init[2:(number_of_features + 1),]
  gam_parameters = list(theta_pair=theta_pair, theta_singleton=theta_singleton, theta=theta)


  phi_placeholder <- initialize_phi(3, number_of_dimensions) 

  pairwise_binary_outliers_train <- extract_pairwise_observed_labels(binary_outliers_train)

  #Randomly shuffle the data
  random_shuffling_indices <- sample(nrow(feat_train))
  feat_train_shuff <- feat_train[random_shuffling_indices,]
  binary_outliers_train_shuff <- binary_outliers_train[random_shuffling_indices,]
  pairwise_binary_outliers_train_shuff <- pairwise_binary_outliers_train[random_shuffling_indices,]

  #Create nfolds equally size folds
  folds <- cut(seq(1,nrow(feat_train_shuff)),breaks=nfolds,labels=FALSE)

  avg_aucs <- c()
  for (cost_iter in 1:length(costs)) {
    lambda <- costs[cost_iter]
    #Perform nfolds-fold cross validation
    aucs <- c()
    for(i in 1:nfolds){
        #Segement your data by fold using the which() function 
        testIndexes <- which(folds==i,arr.ind=TRUE)
        feat_test_fold <- feat_train_shuff[testIndexes,]
        outliers_test_fold <- binary_outliers_train_shuff[testIndexes,]
        pairwise_outliers_test_fold <- pairwise_binary_outliers_train_shuff[testIndexes,]
        feat_train_fold <- feat_train_shuff[-testIndexes,]
        outliers_train_fold <- binary_outliers_train_shuff[-testIndexes,]
        pairwise_outliers_train_fold <- pairwise_binary_outliers_train_shuff[-testIndexes,]

        # num_grad <- grad(compute_exact_crf_likelihood_for_lbfgs, gradient_variable_vec, feat=feat_test_fold, discrete_outliers=outliers_test_fold, posterior=outliers_test_fold, posterior_pairwise=pairwise_outliers_test_fold, phi=phi_placeholder, lambda=lambda, lambda_pair=0, lambda_singleton=0, independent_variables=independent_variables)
        # actual_grad <- compute_exact_crf_gradient_for_lbfgs(gradient_variable_vec, feat=feat_test_fold, discrete_outliers=outliers_test_fold, posterior=outliers_test_fold, posterior_pairwise=pairwise_outliers_test_fold, phi=phi_placeholder, lambda=lambda, lambda_pair=0, lambda_singleton=0, independent_variables=independent_variables)



      lbfgs_output <- lbfgs(compute_exact_crf_likelihood_for_lbfgs, compute_exact_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat_train_fold, discrete_outliers=outliers_train_fold, posterior=outliers_train_fold, posterior_pairwise=pairwise_outliers_train_fold, phi=phi_placeholder, lambda=lambda, lambda_pair=0, lambda_singleton=0, independent_variables=independent_variables,invisible=1)
      # Check to make sure LBFGS converged OK
      if (lbfgs_output$convergence != 0) {
        print(paste0("LBFGS optimazation on CRF did not converge. It reported convergence error of: ", lbfgs_output$convergence))
        print(lbfgs_output$message)
      }


      # Get optimized crf coefficients back into model_params format
      gam_parameters$theta_singleton <- lbfgs_output$par[1:number_of_dimensions]
      for (dimension in 1:number_of_dimensions) {
        gam_parameters$theta[,dimension] <- lbfgs_output$par[(number_of_dimensions + 1 + ncol(feat_train)*(dimension-1)):(number_of_dimensions + ncol(feat_train)*(dimension))]
      }
      #gam_parameters$theta_pair[1,] <- lbfgs_output$par[(number_of_dimensions + (number_of_dimensions*ncol(feat_train)) + 1):length(lbfgs_output$par)]
      #theta_pair[1,] <- x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)]
      gam_parameters$theta_pair <- matrix(lbfgs_output$par[(number_of_dimensions + (number_of_dimensions*ncol(feat_train)) + 1):length(lbfgs_output$par)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

      # Compute expected value of the CRFs (mu)
      mu_list_test <- update_marginal_probabilities_exact_inference_cpp(feat_test_fold, outliers_test_fold, gam_parameters$theta_singleton, gam_parameters$theta_pair, gam_parameters$theta, phi_placeholder$inlier_component, phi_placeholder$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
      mu_test <- mu_list_test$probability

      # Compute roc
      test_predictions <- as.vector(mu_test)
      test_labels <- as.vector(outliers_test_fold)
      roc_obj <- roc.curve(scores.class0 = test_predictions[test_labels==1], scores.class1 = test_predictions[test_labels==0], curve = T)
      auc <- roc_obj$auc
      aucs <- c(aucs, auc)
    }
    avg_aucs <- c(avg_aucs, mean(aucs))
  }
  # Get best (one with highest avg auc across folds) lambda 
  print(avg_aucs)
  best_index <- which(avg_aucs==max(avg_aucs))
  best_lambda <- costs[best_index]
  # Using best lambda, recompute GAM
  lbfgs_output <- lbfgs(compute_exact_crf_likelihood_for_lbfgs, compute_exact_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat_train_shuff, discrete_outliers=binary_outliers_train_shuff, posterior=binary_outliers_train_shuff, posterior_pairwise=pairwise_binary_outliers_train_shuff, phi=phi_placeholder, lambda=best_lambda, lambda_pair=0, lambda_singleton=0, independent_variables=independent_variables,invisible=1)
  # Get optimized crf coefficients back into model_params format
  gam_parameters$theta_singleton <- lbfgs_output$par[1:number_of_dimensions]
  for (dimension in 1:number_of_dimensions) {
    gam_parameters$theta[,dimension] <- lbfgs_output$par[(number_of_dimensions + 1 + ncol(feat_train)*(dimension-1)):(number_of_dimensions + ncol(feat_train)*(dimension))]
  }
  #gam_parameters$theta_pair[1,] <- lbfgs_output$par[(number_of_dimensions + (number_of_dimensions*ncol(feat_train)) + 1):length(lbfgs_output$par)]
  gam_parameters$theta_pair <- matrix(lbfgs_output$par[(number_of_dimensions + (number_of_dimensions*ncol(feat_train)) + 1):length(lbfgs_output$par)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

  return(list(lambda=best_lambda, gam_parameters=gam_parameters))
}


fully_observed_gam_outlier_scatterplot_comparison_in_one_dimension <- function(df, outlier_type) {
   #PLOT!
  scatter <-  ggplot(df,aes(coloring,gam)) + geom_point(size=.1,aes(colour=watershed)) + theme(text = element_text(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) 
  scatter <- scatter + scale_color_gradient(low="pink",high="blue")
  scatter <- scatter +  labs(colour="Watershed posterior",x = "mean[-log10(outlier pvalue)]", y = "GAM posterior", title=outlier_type) 
  scatter <- scatter + theme(legend.position="right")
  return(scatter)
}

# Limit to fully observed cases
# Have one plot per outlier type showing:
### x-axis median outlier pvalue across observed tissues
### Y-axis GAM posterior
### Colored by watershed posterior
########################################
fully_observed_gam_outlier_scatterplot_comparison_colored_by_watershed <- function(data, output_file) {
  options(bitmapType = 'cairo', device = 'pdf')
  # Remove instances that do not have all 3 expression signals
  fully_observed_indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  fully_observed_data <- data[fully_observed_indices,]
  # Remove directionality/sign from total expression pvalues
  fully_observed_data$total_expression_outlier_pvalue <- abs(fully_observed_data$total_expression_outlier_pvalue)
  # Compute mean outlier score (acrosss all 3 outlier classes)
  outlier_means <- rowMeans(-log10(1e-6 + fully_observed_data[,2:4]))

  # Put each outlier class into it own neat data frame
  splicing_df <- data.frame(gam=fully_observed_data$splicing_gam_crf_posterior,watershed=fully_observed_data$splicing_watershed_posterior, coloring=outlier_means)
  te_df <- data.frame(gam=fully_observed_data$total_expression_gam_crf_posterior,watershed=fully_observed_data$total_expression_watershed_posterior, coloring=outlier_means)
  ase_df <- data.frame(gam=fully_observed_data$ase_gam_crf_posterior,watershed=fully_observed_data$ase_watershed_posterior, coloring=outlier_means)

  splicing_plot <- fully_observed_gam_outlier_scatterplot_comparison_in_one_dimension(splicing_df, "splice")
  te_plot <- fully_observed_gam_outlier_scatterplot_comparison_in_one_dimension(te_df, "total expression")
  ase_plot <- fully_observed_gam_outlier_scatterplot_comparison_in_one_dimension(ase_df, "ase")
  combined_plot <- plot_grid(ase_plot, splicing_plot, te_plot, ncol=1)

  ggsave(combined_plot, file=output_file, width=19,height=20,units="cm")

}

fully_observed_watershed_river_scatterplot_comparison_in_one_dimension <- function(df, outlier_type, coloring_label) {
  spearman_rho = cor(df$river, df$watershed, method="spearman") 
  #PLOT!
  scatter <-  ggplot(df,aes(river,watershed)) + geom_point(size=.1,aes(colour=coloring)) + theme(text = element_text(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) 
  scatter <- scatter + scale_color_gradient(low="pink",high="blue")
  scatter <- scatter +  labs(colour=coloring_label,x = "RIVER posterior", y = "Watershed posterior", title=paste0(outlier_type, " / spearman rho: ", round(spearman_rho,digits=2))) 
  scatter <- scatter + theme(legend.position="right")
  return(scatter)
}

fully_observed_watershed_river_scatterplot_comparison_colored_by_median_outlier_score <- function(data, output_file) {
  options(bitmapType = 'cairo', device = 'pdf')
  # Remove instances that do not have all 3 expression signals
  fully_observed_indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  fully_observed_data <- data[fully_observed_indices,]
  # Remove directionality/sign from total expression pvalues
  fully_observed_data$total_expression_outlier_pvalue <- abs(fully_observed_data$total_expression_outlier_pvalue)
  # Compute mean outlier score (acrosss all 3 outlier classes)
  outlier_means <- rowMeans(-log10(1e-6 + fully_observed_data[,2:4]))

  # Put each outlier class into it own neat data frame
  splicing_df <- data.frame(river=fully_observed_data$splicing_river_posterior,watershed=fully_observed_data$splicing_watershed_posterior, coloring=outlier_means)
  te_df <- data.frame(river=fully_observed_data$total_expression_river_posterior,watershed=fully_observed_data$total_expression_watershed_posterior, coloring=outlier_means)
  ase_df <- data.frame(river=fully_observed_data$ase_river_posterior,watershed=fully_observed_data$ase_watershed_posterior, coloring=outlier_means)

  splicing_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(splicing_df, "splice", "mean[-log10(pvalue)]")
  te_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(te_df, "total expression", "mean[-log10(pvalue)]")
  ase_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(ase_df, "ase", "mean[-log10(pvalue)]")

  combined_plot <- plot_grid(ase_plot, splicing_plot, te_plot, ncol=1)

  ggsave(combined_plot, file=output_file, width=19,height=20,units="cm")

}

fully_observed_watershed_river_scatterplot_comparison_colored_by_average_outlier_score <- function(data, output_file) {
  options(bitmapType = 'cairo', device = 'pdf')
  # Remove instances that do not have all 3 expression signals
  fully_observed_indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  fully_observed_data <- data[fully_observed_indices,]
  # Remove directionality/sign from total expression pvalues
  fully_observed_data$total_expression_outlier_pvalue <- abs(fully_observed_data$total_expression_outlier_pvalue)
  # Compute mean outlier score (acrosss all 3 outlier classes)
  outlier_medians <- rowMedians(as.matrix(-log10(1e-6 + fully_observed_data[,2:4])))

  # Put each outlier class into it own neat data frame
  splicing_df <- data.frame(river=fully_observed_data$splicing_river_posterior,watershed=fully_observed_data$splicing_watershed_posterior, coloring=outlier_medians)
  te_df <- data.frame(river=fully_observed_data$total_expression_river_posterior,watershed=fully_observed_data$total_expression_watershed_posterior, coloring=outlier_medians)
  ase_df <- data.frame(river=fully_observed_data$ase_river_posterior,watershed=fully_observed_data$ase_watershed_posterior, coloring=outlier_medians)

  splicing_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(splicing_df, "splice", "median[-log10(pvalue)]")
  te_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(te_df, "total expression", "median[-log10(pvalue)]")
  ase_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(ase_df, "ase", "median[-log10(pvalue)]")

  combined_plot <- plot_grid(ase_plot, splicing_plot, te_plot, ncol=1)

  ggsave(combined_plot, file=output_file, width=19,height=20,units="cm")

}

fully_observed_watershed_river_scatterplot_comparison_colored_by_classes_outlier_score <- function(data, output_file) {
  options(bitmapType = 'cairo', device = 'pdf')
  # Remove instances that do not have all 3 expression signals
  fully_observed_indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  fully_observed_data <- data[fully_observed_indices,]
  # Remove directionality/sign from total expression pvalues
  fully_observed_data$total_expression_outlier_pvalue <- abs(fully_observed_data$total_expression_outlier_pvalue)



  # Put each outlier class into it own neat data frame
  splicing_df <- data.frame(river=fully_observed_data$splicing_river_posterior,watershed=fully_observed_data$splicing_watershed_posterior, coloring=-log10(1e-6 + fully_observed_data$splicing_outlier_pvalue))
  te_df <- data.frame(river=fully_observed_data$total_expression_river_posterior,watershed=fully_observed_data$total_expression_watershed_posterior, coloring=-log10(1e-6 + fully_observed_data$total_expression_outlier_pvalue))
  ase_df <- data.frame(river=fully_observed_data$ase_river_posterior,watershed=fully_observed_data$ase_watershed_posterior, coloring=-log10(1e-6 + fully_observed_data$ase_outlier_pvalue))

  splicing_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(splicing_df, "splice", "-log10(pvalue)")
  te_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(te_df, "total expression", "-log10(pvalue)")
  ase_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(ase_df, "ase", "-log10(pvalue)")

  combined_plot <- plot_grid(ase_plot, splicing_plot, te_plot, ncol=1)

  ggsave(combined_plot, file=output_file, width=19,height=20,units="cm")

}

fully_observed_watershed_river_scatterplot_comparison_colored_by_classes_independent_gam_score <- function(data, output_file) {
  options(bitmapType = 'cairo', device = 'pdf')
  # Remove instances that do not have all 3 expression signals
  fully_observed_indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  fully_observed_data <- data[fully_observed_indices,]
  # Remove directionality/sign from total expression pvalues
  fully_observed_data$total_expression_outlier_pvalue <- abs(fully_observed_data$total_expression_outlier_pvalue)



  # Put each outlier class into it own neat data frame
  splicing_df <- data.frame(river=fully_observed_data$splicing_river_posterior,watershed=fully_observed_data$splicing_watershed_posterior, coloring=fully_observed_data$splicing_gam_posterior)
  te_df <- data.frame(river=fully_observed_data$total_expression_river_posterior,watershed=fully_observed_data$total_expression_watershed_posterior, coloring=fully_observed_data$total_expression_gam_posterior)
  ase_df <- data.frame(river=fully_observed_data$ase_river_posterior,watershed=fully_observed_data$ase_watershed_posterior, coloring=fully_observed_data$ase_gam_posterior)

  splicing_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(splicing_df, "splice", "GAM posterior")
  te_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(te_df, "total expression", "GAM posterior")
  ase_plot <- fully_observed_watershed_river_scatterplot_comparison_in_one_dimension(ase_df, "ase", "GAM posterior")

  combined_plot <- plot_grid(ase_plot, splicing_plot, te_plot, ncol=1)

  ggsave(combined_plot, file=output_file, width=19,height=20,units="cm")

}


visualize_watershed_posterior_distributions <- function(data, output_file) {
  options(bitmapType = 'cairo', device = 'pdf')
  thresh=.5
  #################################################
  #### First part is to put data in correct format
  #################################################
  # Initialize arrays
  posteriors <- c()
  outlier_types <- c()
  observed_types <- c()
  # Fully observed case
  indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,s,te",3))
  # TE unobserved
  indices <- !is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,s",3))

  # ase unobserved
  indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("s,te",3))
  # splice unobserved
  indices <- is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,te",3))
  # ase observed
  indices <- is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase",3))
  # splicing observed
  indices <- !is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("s",3))
  # TE observed
  indices <- is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("te",3))

  df <- data.frame(posterior=posteriors, outlier_type=factor(outlier_types,levels=c("ase","splice","total expression")), observed_type=factor(observed_types, levels=c("ase","s","te","ase,s","s,te","ase,te","ase,s,te")))

  p_5 <- ggplot(data=df, aes(x=observed_type, y=posterior, fill=outlier_type)) +
        geom_bar(stat="identity", color="black", position=position_dodge())+
        theme(text = element_text(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) +
        labs(x="Observed outlier type", y=paste0("% posterior >= ", thresh), fill="Outlier type")

  thresh=.7
  #################################################
  #### First part is to put data in correct format
  #################################################
  # Initialize arrays
  posteriors <- c()
  outlier_types <- c()
  observed_types <- c()
  # Fully observed case
  indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,s,te",3))
  # TE unobserved
  indices <- !is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,s",3))

  # ase unobserved
  indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("s,te",3))
  # splice unobserved
  indices <- is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,te",3))
  # ase observed
  indices <- is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase",3))
  # splicing observed
  indices <- !is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("s",3))
  # TE observed
  indices <- is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("te",3))

  df <- data.frame(posterior=posteriors, outlier_type=factor(outlier_types,levels=c("ase","splice","total expression")), observed_type=factor(observed_types, levels=c("ase","s","te","ase,s","s,te","ase,te","ase,s,te")))

  p_7 <- ggplot(data=df, aes(x=observed_type, y=posterior, fill=outlier_type)) +
        geom_bar(stat="identity", color="black", position=position_dodge())+
        theme(text = element_text(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) +
        labs(x="Observed outlier type", y=paste0("% posterior >= ", thresh), fill="Outlier type")

  thresh=.9
  #################################################
  #### First part is to put data in correct format
  #################################################
  # Initialize arrays
  posteriors <- c()
  outlier_types <- c()
  observed_types <- c()
  # Fully observed case
  indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,s,te", 3))
  # TE unobserved
  indices <- !is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,s",3))

  # ase unobserved
  indices <- !is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("s,te",3))
  # splice unobserved
  indices <- is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase,te",3))
  # ase observed
  indices <- is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & !is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("ase",3))
  # splicing observed
  indices <- !is.nan(data$splicing_outlier_pvalue) & is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("s",3))
  # TE observed
  indices <- is.nan(data$splicing_outlier_pvalue) & !is.nan(data$total_expression_outlier_pvalue) & is.nan(data$ase_outlier_pvalue)
  posteriors <- c(posteriors, sum(data[indices,]$splicing_watershed_posterior >= thresh)/sum(indices) , sum(data[indices,]$total_expression_watershed_posterior >= thresh)/sum(indices), sum(data[indices,]$ase_watershed_posterior >= thresh)/sum(indices))
  outlier_types <- c(outlier_types, "splice", "total expression", "ase")
  observed_types <- c(observed_types, rep("te",3))

  df <- data.frame(posterior=posteriors, outlier_type=factor(outlier_types,levels=c("ase","splice","total expression")), observed_type=factor(observed_types, levels=c("ase","s","te","ase,s","s,te","ase,te","ase,s,te")))
  p_9 <- ggplot(data=df, aes(x=observed_type, y=posterior, fill=outlier_type)) +
        geom_bar(stat="identity", color="black", position=position_dodge())+
        theme(text = element_text(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) +
        labs(x="Observed outlier type", y=paste0("% posterior >= ", thresh), fill="Outlier type")


  combined <- plot_grid(p_5,p_7,p_9,ncol=1)

  ggsave(combined, file=output_file, width=19,height=20,units="cm")



}

missing_gam_outlier_scatterplot_in_one_dimension <- function(df, outlier_type) {
  #PLOT!
  scatter <-  ggplot(df,aes(gam,outlier)) + geom_point(size=.1,aes(colour=watershed)) + theme(text = element_text(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) 
  scatter <- scatter + scale_color_gradient(low="pink",high="blue")
  scatter <- scatter +  labs(x = "GAM posterior", y = "mean[-log10(outlier)]", colour="Watershed posterior", title=outlier_type) 
  scatter <- scatter + theme(legend.position="right")
  return(scatter)
}


# Limit to missing cases for each outlier type
# Have one plot per outlier type showing:
### x-axis median outlier pvalue across observed tisssues
### Y-axis GAM Posterior
### Colored by watershed score
########################################
missing_gam_outlier_scatterplot_comparison_colored_by_watershed_score <- function(data, output_file) {
   options(bitmapType = 'cairo', device = 'pdf')

  data$total_expression_outlier_pvalue <- abs(data$total_expression_outlier_pvalue)

  outlier_means <- rowMeans(-log10(1e-6 + data[,2:4]), na.rm=TRUE)



  missing_splice_indices <- is.nan(data$splicing_outlier_pvalue)
  missing_ase_indices <- is.nan(data$ase_outlier_pvalue)
  missing_te_indices <- is.nan(data$total_expression_outlier_pvalue)


  te_df <- data.frame(watershed=data[missing_te_indices,]$total_expression_watershed_posterior, gam=data[missing_te_indices,]$total_expression_gam_crf_posterior, outlier=outlier_means[missing_te_indices])
  ase_df <- data.frame(watershed=data[missing_ase_indices,]$ase_watershed_posterior, gam=data[missing_ase_indices,]$ase_gam_crf_posterior, outlier=outlier_means[missing_ase_indices])
  splice_df <- data.frame(watershed=data[missing_splice_indices,]$splicing_watershed_posterior, gam=data[missing_splice_indices,]$splicing_gam_crf_posterior, outlier=outlier_means[missing_splice_indices])


  ase_plot <- missing_gam_outlier_scatterplot_in_one_dimension(ase_df, "ase")
  te_plot <- missing_gam_outlier_scatterplot_in_one_dimension(te_df, "total expression")
  splice_plot <- missing_gam_outlier_scatterplot_in_one_dimension(splice_df, "splice")

  combined_plot <- plot_grid(ase_plot, splice_plot, te_plot, ncol=1)


  ggsave(combined_plot, file=output_file, width=19,height=20,units="cm")

}



######################
# Command Line args
######################
pvalue_fraction <- as.numeric(args[1])
number_of_dimensions <- as.numeric(args[2])
inference_method <- args[3]
pseudocount <- as.numeric(args[4])
fully_observed_input_file <- args[5]
all_variants_input_file <- args[6]
all_variants_variant_level_input_file <- args[7]
output_stem <- args[8]

#######################################
## Load in data
#######################################
# Load in data for all variants
# This is the data we are making predictions on
predictions_data <- load_watershed_data(all_variants_input_file, number_of_dimensions, pvalue_fraction)
mean_feat <- apply(predictions_data$feat, 2, mean)
sd_feat <- apply(predictions_data$feat, 2, sd)
predictions_feat <- scale(predictions_data$feat, center=mean_feat, scale=sd_feat) 
predictions_discretized_outliers <- predictions_data$outliers_discrete
predictions_pvalues_outliers <- predictions_data$outlier_pvalues

# Load in data for all variants at variant level
# This is more data that we are making predictions on 
predictions_data_variant_level <- load_watershed_data(all_variants_variant_level_input_file, number_of_dimensions, pvalue_fraction)  
predictions_variant_level_feat <- scale(predictions_data_variant_level$feat, center=mean_feat, scale=sd_feat)
predictions_variant_level_discretized_outliers <- predictions_data_variant_level$outliers_discrete
predictions_variant_level_pvalues_outliers <- predictions_data_variant_level$outlier_pvalues

# Load in fully observed data. Note that pvalue fraction is just here as  dummy variable (as we do not use binary outlier calls for this analysis)
# This is the data we are training on
training_data <- load_watershed_data(fully_observed_input_file, number_of_dimensions, pvalue_fraction)  
training_feat <- scale(training_data$feat, center=mean_feat, scale=sd_feat)
training_discretized_outliers <- training_data$outliers_discrete
training_binary_outliers <- training_data$outliers_binary


#######################################
## Train Watershed model on fully observed training data
#######################################
phi_init <- initialize_phi(3, number_of_dimensions) 
costs= c(.1, .01, 1e-3, 1e-4)
nfolds <- 4
lambda_singleton <- 0
lambda_pair <- 0

if (FALSE) {
independent_variables="false"
tied_gam_data <- genomic_annotation_model_cv(training_feat, training_binary_outliers, nfolds, costs, independent_variables)
watershed_model <- integratedEM(training_feat, training_discretized_outliers, phi_init, tied_gam_data$gam_parameters$theta_pair, tied_gam_data$gam_parameters$theta_singleton, tied_gam_data$gam_parameters$theta, pseudocount, tied_gam_data$lambda, lambda_singleton, lambda_pair, number_of_dimensions, inference_method, independent_variables)
saveRDS(watershed_model, file=paste0(output_stem, "_watershed_model.rds"))
saveRDS(tied_gam_data, file=paste0(output_stem, "_tied_gam_model.rds"))

independent_variables="true"
gam_data <- genomic_annotation_model_cv(training_feat, training_binary_outliers, nfolds, costs, independent_variables)
river_model <- integratedEM(training_feat, training_discretized_outliers, phi_init, gam_data$gam_parameters$theta_pair, gam_data$gam_parameters$theta_singleton, gam_data$gam_parameters$theta, pseudocount, gam_data$lambda, lambda_singleton, lambda_pair, number_of_dimensions, inference_method, independent_variables)
saveRDS(river_model, file=paste0(output_stem, "_river_model.rds"))
saveRDS(gam_data, file=paste0(output_stem, "_independent_gam_model.rds"))


#######################################
## Compute posterior probabilities based on fitted models
#######################################
# Watershed model
watershed_posterior_list <- update_marginal_probabilities_exact_inference_cpp(predictions_feat, predictions_discretized_outliers, watershed_model$theta_singleton, watershed_model$theta_pair, watershed_model$theta, watershed_model$phi$inlier_component, watershed_model$phi$outlier_component, watershed_model$number_of_dimensions, choose(watershed_model$number_of_dimensions, 2), TRUE)
watershed_posteriors <- watershed_posterior_list$probability
# Watershed-independent model
river_posterior_list <- update_marginal_probabilities_exact_inference_cpp(predictions_feat, predictions_discretized_outliers, river_model$theta_singleton, river_model$theta_pair, river_model$theta, river_model$phi$inlier_component, river_model$phi$outlier_component, river_model$number_of_dimensions, choose(river_model$number_of_dimensions, 2), TRUE)
river_posteriors <- river_posterior_list$probability

# Tied GAM model
tied_gam_posterior_test <- update_marginal_probabilities_exact_inference_cpp(predictions_feat, predictions_discretized_outliers, tied_gam_data$gam_parameters$theta_singleton, tied_gam_data$gam_parameters$theta_pair, tied_gam_data$gam_parameters$theta, river_model$phi$outlier_component, river_model$phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
tied_gam_posteriors <- tied_gam_posterior_test$probability

# Tied GAM model
gam_posterior_test <- update_marginal_probabilities_exact_inference_cpp(predictions_feat, predictions_discretized_outliers, gam_data$gam_parameters$theta_singleton, gam_data$gam_parameters$theta_pair, gam_data$gam_parameters$theta, river_model$phi$outlier_component, river_model$phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
gam_posteriors <- gam_posterior_test$probability

#######################################
## Save predictions to output file
#######################################

posterior_mat <- cbind(rownames(predictions_feat), predictions_pvalues_outliers, gam_posteriors, tied_gam_posteriors, river_posteriors, watershed_posteriors)
colnames(posterior_mat) = c("sample_names","splicing_outlier_pvalue", "total_expression_outlier_pvalue", "ase_outlier_pvalue", "splicing_gam_posterior", "total_expression_gam_posterior", "ase_gam_posterior", "splicing_gam_crf_posterior", "total_expression_gam_crf_posterior", "ase_gam_crf_posterior", "splicing_river_posterior", "total_expression_river_posterior", "ase_river_posterior", "splicing_watershed_posterior", "total_expression_watershed_posterior", "ase_watershed_posterior")

write.table(posterior_mat,file=paste0(output_stem,"_posteriors.txt"), sep="\t", quote=FALSE, row.names=FALSE)
}
#######################################
## Compute posterior probabilities for variant level samples based on fitted models
#######################################
#
watershed_model <- readRDS(paste0(output_stem, "_watershed_model.rds"))
tied_gam_data <- readRDS(paste0(output_stem, "_tied_gam_model.rds"))
river_model <- readRDS(paste0(output_stem, "_river_model.rds"))
gam_data <- readRDS(paste0(output_stem, "_independent_gam_model.rds"))
#
# Watershed model
watershed_posterior_list <- update_marginal_probabilities_exact_inference_cpp(predictions_variant_level_feat, predictions_variant_level_discretized_outliers, watershed_model$theta_singleton, watershed_model$theta_pair, watershed_model$theta, watershed_model$phi$inlier_component, watershed_model$phi$outlier_component, watershed_model$number_of_dimensions, choose(watershed_model$number_of_dimensions, 2), TRUE)
watershed_posteriors <- watershed_posterior_list$probability
# Watershed-independent model
river_posterior_list <- update_marginal_probabilities_exact_inference_cpp(predictions_variant_level_feat, predictions_variant_level_discretized_outliers, river_model$theta_singleton, river_model$theta_pair, river_model$theta, river_model$phi$inlier_component, river_model$phi$outlier_component, river_model$number_of_dimensions, choose(river_model$number_of_dimensions, 2), TRUE)
river_posteriors <- river_posterior_list$probability

# Tied GAM model
tied_gam_posterior_test <- update_marginal_probabilities_exact_inference_cpp(predictions_variant_level_feat, predictions_variant_level_discretized_outliers, tied_gam_data$gam_parameters$theta_singleton, tied_gam_data$gam_parameters$theta_pair, tied_gam_data$gam_parameters$theta, river_model$phi$outlier_component, river_model$phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
tied_gam_posteriors <- tied_gam_posterior_test$probability

# Tied GAM model
gam_posterior_test <- update_marginal_probabilities_exact_inference_cpp(predictions_variant_level_feat, predictions_variant_level_discretized_outliers, gam_data$gam_parameters$theta_singleton, gam_data$gam_parameters$theta_pair, gam_data$gam_parameters$theta, river_model$phi$outlier_component, river_model$phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
gam_posteriors <- gam_posterior_test$probability

#######################################
## Save predictions to output file
#######################################

posterior_mat <- cbind(rownames(predictions_variant_level_feat), predictions_variant_level_pvalues_outliers, gam_posteriors, tied_gam_posteriors, river_posteriors, watershed_posteriors)
colnames(posterior_mat) = c("sample_names","splicing_outlier_pvalue", "total_expression_outlier_pvalue", "ase_outlier_pvalue", "splicing_gam_posterior", "total_expression_gam_posterior", "ase_gam_posterior", "splicing_gam_crf_posterior", "total_expression_gam_crf_posterior", "ase_gam_crf_posterior", "splicing_river_posterior", "total_expression_river_posterior", "ase_river_posterior", "splicing_watershed_posterior", "total_expression_watershed_posterior", "ase_watershed_posterior")

write.table(posterior_mat,file=paste0(output_stem,"_variant_level_posteriors.txt"), sep="\t", quote=FALSE, row.names=FALSE)


if (FALSE) {
#######################################
## Visualize Predictions
#######################################

data <- read.table(paste0(output_stem,"_posteriors.txt"), header=TRUE)



# Look at distribution of watershed posteriors depending on:
#### Which outlier signals are observed (x-axis)
#### which outlier posterior we are looking at (y-axis)
########################################
output_file <- paste0(output_stem, "_watershed_posterior_distributions.pdf")
#visualize_watershed_posterior_distributions(data, output_file)


# Limit to fully observed cases
# Have one plot per outlier type showing:
### x-axis River score
### Y-axis watershed score
### Colored by classes independent-GAM score
########################################
output_file <- paste0(output_stem, "_fully_observed_watershed_river_scatterplot_comparison_colored_by_classes_independent_gam_score.pdf")
#fully_observed_watershed_river_scatterplot_comparison_colored_by_classes_independent_gam_score(data, output_file)

# Limit to fully observed cases
# Have one plot per outlier type showing:
### x-axis River score
### Y-axis watershed score
### Colored by average outlier score
########################################
output_file <- paste0(output_stem, "_fully_observed_watershed_river_scatterplot_comparison_colored_by_classes_outlier_score.pdf")
#fully_observed_watershed_river_scatterplot_comparison_colored_by_classes_outlier_score(data, output_file)

# Limit to fully observed cases
# Have one plot per outlier type showing:
### x-axis River score
### Y-axis watershed score
### Colored by median outlier score
########################################
output_file <- paste0(output_stem, "_fully_observed_watershed_river_scatterplot_comparison_colored_by_median_outlier_score.pdf")
#fully_observed_watershed_river_scatterplot_comparison_colored_by_median_outlier_score(data, output_file)

# Limit to fully observed cases
# Have one plot per outlier type showing:
### x-axis River score
### Y-axis watershed score
### Colored by classes outlier score
########################################
output_file <- paste0(output_stem, "_fully_observed_watershed_river_scatterplot_comparison_colored_by_average_outlier_score.pdf")
#fully_observed_watershed_river_scatterplot_comparison_colored_by_average_outlier_score(data, output_file)


# Limit to fully observed cases
# Have one plot per outlier type showing:
### x-axis median outlier pvalue across observed tissues
### Y-axis GAM posterior
### Colored by watershed posterior
########################################
output_file <- paste0(output_stem, "_fully_observed_gam_outlier_scatterplot_comparison_colored_by_watershed_score.pdf")
fully_observed_gam_outlier_scatterplot_comparison_colored_by_watershed(data, output_file)

# Limit to missing cases for each outlier type
# Have one plot per outlier type showing:
### x-axis median outlier pvalue across observed tisssues
### Y-axis GAM Posterior
### Colored by watershed score
########################################
output_file <- paste0(output_stem, "_missing_gam_outlier_scatterplot_comparison_colored_by_watershed score.pdf")
missing_gam_outlier_scatterplot_comparison_colored_by_watershed_score(data, output_file)

}