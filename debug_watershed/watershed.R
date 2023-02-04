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
library(RColorBrewer)
library(DirichletReg)
sourceCpp("crf_exact_updates.cpp")
sourceCpp("crf_variational_updates.cpp")
sourceCpp("independent_crf_exact_updates.cpp")

initialize_genomic_annotation_variables_v_rand <- function(number_of_features, number_of_dimensions, independent_variables) {
  if (independent_variables == "true") {
    pair_value = 0
    theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
  } else if (independent_variables == "false") {
    pair_value = 4
    theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
  } else if (independent_variables == "false_geno") {
    pair_value = 1e-7
    theta_pair = matrix(0, number_of_features+1, choose(number_of_dimensions, 2))
    theta_pair[1,] <- numeric(choose(number_of_dimensions, 2)) + pair_value
  }

  beta_init = matrix(rnorm((number_of_features+1)*number_of_dimensions,sd=.1),number_of_features+1, number_of_dimensions)
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

 
grad_desc=function(grad_fxn, log_likelihood_fxn, x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables, convergence_criteria, master_stepsize, iter) {
  
  output_root <- paste0("/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/unsupervised_modeling/watershed_tbt_debug/iter_", iter,"_")

  saveRDS(x, paste0(output_root, "x.rds"))
  saveRDS(feat, paste0(output_root, "feat.rds"))
  saveRDS(discrete_outliers, paste0(output_root, "discrete_outliers.rds"))
  saveRDS(posterior, paste0(output_root, "posterior.rds"))
  saveRDS(posterior_pairwise, paste0(output_root, "posterior_pairwise.rds"))
  saveRDS(phi, paste0(output_root, "phi.rds"))
  saveRDS(mu_init, paste0(output_root, "mu_init.rds"))
  saveRDS(mu_pairwise_init, paste0(output_root, "mu_pairwise_init.rds"))


  convergence_value = 0
  convergence_message = "no errors"

  number_of_dimensions <- dim(discrete_outliers)[2]
  num_genomic_features <- dim(feat)[2]

  # Get crf coefficients back into inference format
  theta_singleton <- x[1:number_of_dimensions]
  theta <- matrix(0,num_genomic_features,number_of_dimensions)
  for (dimension in 1:number_of_dimensions) {
	theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
  }
  theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)
  mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, mu_init, FALSE)
  mu <- mu_list$probability
  mu_pairwise <- mu_list$probability_pairwise

  convergence = FALSE
  iterations = 1
  progress=list()
  prev_likelihood = log_likelihood_fxn(x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, mu_init=mu, mu_pairwise_init=mu_pairwise, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
  print(prev_likelihood)
  x_init = x
  while(convergence==FALSE) {
    g <- grad_fxn(x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, mu_init=mu, mu_pairwise_init=mu_pairwise, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
    x = x-master_stepsize*g
    

  	# Get crf coefficients back into inference format
    theta_singleton <- x[1:number_of_dimensions]
    theta <- matrix(0,num_genomic_features,number_of_dimensions)
    for (dimension in 1:number_of_dimensions) {
	   theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
     }
     theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)
     mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, mu, FALSE)
     mu <- mu_list$probability
     mu_pairwise <- mu_list$probability_pairwise


    likelihood <- log_likelihood_fxn(x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, mu_init=mu, mu_pairwise_init=mu_pairwise, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
    print(likelihood)
    if (abs(prev_likelihood - likelihood) < convergence_criteria) {
    	convergence = TRUE
      if (iterations == 1) {
        x = x_init
        convergence_value = 2
        convergence_message = "The initial variables already minimize the objective function."
      }
    }
    prev_likelihood = likelihood
    iterations = iterations + 1
  }
  list(par=x,log_prob=progress,convergence=convergence_value, message=convergence_message)
}


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

initialize_phi_tbt<- function(num_bins,dim) {
  phi_outlier <- matrix(1,dim,num_bins)
  phi_inlier <- matrix(1,dim,num_bins)
  phi_inlier[,1] = .895
  phi_inlier[,2] = .10
  phi_inlier[,3] = .005
 

  phi_outlier[,1] = .25
  phi_outlier[,2] = .35
  phi_outlier[,3] = .4

  phi_init <- list(inlier_component = phi_inlier, outlier_component = phi_outlier)
  return(phi_init)
}

initialize_phi_tbt_te<- function(num_bins,dim) {
  phi_outlier <- matrix(1,dim,num_bins)
  phi_inlier <- matrix(1,dim,num_bins)
  phi_inlier[,1] = .05
  phi_inlier[,2] = .9
  phi_inlier[,3] = .05
 

  phi_outlier[,1] = .49
  phi_outlier[,2] = .02
  phi_outlier[,3] = .49

  phi_init <- list(inlier_component = phi_inlier, outlier_component = phi_outlier)
  return(phi_init)
}



initialize_genomic_annotation_variables <- function(number_of_features, number_of_dimensions, independent_variables, theta_pair_init) {
	if (independent_variables == "true") {
		pair_value = 0
		theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
	} else if (independent_variables == "false") {
		pair_value = theta_pair_init
		theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
	} else if (independent_variables == "false_geno") {
		pair_value = theta_pair_init
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

compute_logistic_regression_likelihood <- function(x, y, feat, lambda) {
	intercept <- x[1]
	theta <- x[2:length(x)]
	log_likelihood <- compute_logistic_regression_likelihood_exact_inference_cpp(y, feat, intercept, theta, lambda)
	return(-log_likelihood)
}

# Calculate gradient of crf likelihood (fxn formatted to be used in LBFGS)
compute_logistic_regression_gradient <- function(x, y, feat, lambda) {
	intercept <- x[1]
	theta <- x[2:length(x)]

	predictions <- logistic_regression_predictions(feat, intercept, theta)

	# Gradient of singleton terms (intercepts)
	grad_singleton <- (colSums(y) - colSums(predictions))*(1/nrow(y))

	# Gradient of theta terms (betas)
	grad_theta <- c()
	dimension <- 1
	temp_grad <- colSums(y[,dimension]*feat) - colSums(predictions[,dimension]*feat)
	grad_theta <- c(grad_theta, temp_grad)

	grad_theta <- grad_theta*(1/nrow(y)) - lambda*theta

	grad <- c(grad_singleton, grad_theta)
	return(-grad)
}


logistic_regression_genomic_annotation_model_cv <- function(feat_train, binary_outliers_train, nfolds, lambda_costs, lambda_pair_costs, independent_variables, inference_method, theta_pair_init, seed_number, lambda_inity) {
	number_of_dimensions <- dim(binary_outliers_train)[2]
	number_of_features <- dim(feat_train)[2]


  gradient_variable_vec <- rep(0, number_of_features+1)
	
	# gam_parameters = list(theta_pair=theta_pair, theta_singleton=theta_singleton, theta=theta)

	#Randomly shuffle the data
	set.seed(seed_number)
  random_shuffling_indices <- sample(nrow(feat_train))
	feat_train_shuff <- feat_train[random_shuffling_indices,]
	binary_outliers_train_shuff <- binary_outliers_train[random_shuffling_indices,]

	#Create nfolds equally size folds
	folds <- cut(seq(1,nrow(feat_train_shuff)),breaks=nfolds,labels=FALSE)

  if (FALSE) {
	avg_aucs <- c()
	for (cost_iter in 1:length(lambda_costs)) {
		lambda <- lambda_costs[cost_iter]
    	lambda_pair <- lambda_pair_costs[cost_iter]
    	print(lambda)
		#Perform nfolds-fold cross validation
		aucs <- c()
		for(i in 1:nfolds){
			pos <- c()
			neg <- c()
    		#Segement your data by fold using the which() function 
    		testIndexes <- which(folds==i,arr.ind=TRUE)
    		feat_test_fold <- feat_train_shuff[testIndexes,]
    		outliers_test_fold <- binary_outliers_train_shuff[testIndexes,]
    		feat_train_fold <- feat_train_shuff[-testIndexes,]
    		outliers_train_fold <- binary_outliers_train_shuff[-testIndexes,]
    		# Perform logistic regression in each tissue seperately
    		for (dimension in 1:number_of_dimensions) {
    			observed_training_indices <- !is.na(outliers_train_fold[,dimension])
    			observed_training_outliers <- as.matrix(outliers_train_fold[observed_training_indices, dimension])
    			observed_training_feat <- feat_train_fold[observed_training_indices,]
    			observed_testing_indices <- !is.na(outliers_test_fold[,dimension])
    			observed_testing_outliers <- outliers_test_fold[observed_testing_indices, dimension]
    			observed_testing_feat <- feat_test_fold[observed_testing_indices,]
    			
    			#log_likelihood <- compute_logistic_regression_likelihood, (gradient_variable_vec, observed_training_outliers, observed_training_feat, lambda)
    			#grad <- compute_logistic_regression_gradient(gradient_variable_vec, observed_training_outliers, observed_training_feat, lambda)
    			lbfgs_output <- lbfgs(compute_logistic_regression_likelihood, compute_logistic_regression_gradient, gradient_variable_vec, y=observed_training_outliers, feat=observed_training_feat, lambda=lambda, invisible=1)
    			 
    			if (lbfgs_output$convergence != 0) {
    				print("ERRROR!")
    				print(lbfgs_output$convergence)
    			}

    			predictions <- c(logistic_regression_predictions(observed_testing_feat, lbfgs_output$par[1], lbfgs_output$par[2:length(lbfgs_output$par)]))
    			pos <- c(pos, predictions[observed_testing_outliers==1])
    			neg <- c(neg, predictions[observed_testing_outliers==0])
    		}


    		pr_obj <- pr.curve(scores.class0=pos, scores.class1=neg,curve=T)
    		auc <- pr_obj$auc.integral
    		print(auc)
    		aucs <- c(aucs, auc)

			#pr_obj <- pr.curve(scores.class0 = test_predictions[test_labels==1], scores.class1 = test_predictions[test_labels==0], curve = T)
			#auc <- pr_obj$auc.integral
			#aucs <- c(aucs, auc)
		}
		avg_aucs <- c(avg_aucs, median(aucs))
	}
	# Get best (one with highest avg auc across folds) lambda 
	print(avg_aucs)
	best_index <- which(avg_aucs==max(avg_aucs))[1]  # [1] for tie breakers
	best_lambda <- lambda_costs[best_index]
  	best_lambda_pair <- lambda_pair_costs[best_index]
	}
  best_lambda <- lambda_inity
  best_lambda_pair <- 0
	# Initialize output variables
  	pair_value = theta_pair_init
  	theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
	beta_init = matrix(0,number_of_features+1, number_of_dimensions)
	theta_singleton = beta_init[1,]
	theta = beta_init[2:(number_of_features + 1),]
	gam_parameters = list(theta_pair=theta_pair, theta_singleton=theta_singleton, theta=theta)

  	for (dimension in 1:number_of_dimensions) {
  		# feat_train_shuff
  		# binary_outliers_train_shuff
  		observed_training_indices <- !is.na(binary_outliers_train_shuff[,dimension])
  		observed_training_outliers <- as.matrix(binary_outliers_train_shuff[observed_training_indices, dimension])
  		observed_training_feat <- feat_train_shuff[observed_training_indices,]

		lbfgs_output <- lbfgs(compute_logistic_regression_likelihood, compute_logistic_regression_gradient, gradient_variable_vec, y=observed_training_outliers, feat=observed_training_feat, lambda=best_lambda, invisible=1)
		if (lbfgs_output$convergence != 0) {
    		print("ERRROR!")
    		print(lbfgs_output$convergence)
    	}
    	gam_parameters$theta_singleton[dimension] <- lbfgs_output$par[1]
    	gam_parameters$theta[,dimension] <- lbfgs_output$par[2:length(lbfgs_output$par)]
  	}
  	print(gam_parameters)


	return(list(lambda=best_lambda, lambda_pair=best_lambda_pair, gam_parameters=gam_parameters))
}



genomic_annotation_model_cv <- function(feat_train, binary_outliers_train, nfolds, lambda_costs, lambda_pair_costs, independent_variables, inference_method, gradient_descent_threshold, theta_pair_init, seed_number) {
  number_of_dimensions <- dim(binary_outliers_train)[2]
	number_of_features <- dim(feat_train)[2]

  	gradient_variable_vec <- initialize_genomic_annotation_variables(number_of_features, number_of_dimensions, independent_variables, theta_pair_init)

  	binary_outliers_train[is.na(binary_outliers_train)] <- 0

	if (independent_variables == "true") {
		pair_value = 0
		theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
	} else if (independent_variables == "false") {
		pair_value = .5
		theta_pair = matrix(pair_value,1, choose(number_of_dimensions, 2))
	} else if (independent_variables == "false_geno") {
		pair_value = .5
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
	for (cost_iter in 1:length(lambda_costs)) {
		lambda <- lambda_costs[cost_iter]
    	lambda_pair <- lambda_pair_costs[cost_iter]
    	print(lambda)
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


    

    	if (inference_method == "exact") {
				lbfgs_output <- lbfgs(compute_exact_crf_likelihood_for_lbfgs, compute_exact_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat_train_fold, discrete_outliers=outliers_train_fold, posterior=outliers_train_fold, posterior_pairwise=pairwise_outliers_train_fold, phi=phi_placeholder, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=0, independent_variables=independent_variables,invisible=1)
		} else if (inference_method == "vi") {
				num_genomic_features <- number_of_features
				theta_singleton <- gradient_variable_vec[1:number_of_dimensions]
				theta <- matrix(0,num_genomic_features,number_of_dimensions)
				for (dimension in 1:number_of_dimensions) {
					theta[,dimension] <- gradient_variable_vec[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
				}
				theta_pair <- matrix(gradient_variable_vec[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(gradient_variable_vec)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)
				mu_init = matrix(.5, dim(feat_train_fold)[1], number_of_dimensions)
       		 	mu_pairwise_init <- matrix(.5, dim(feat_train_fold)[1], choose(number_of_dimensions, 2))

				#mu_list <- update_marginal_probabilities_vi_cpp(feat_train_fold, outliers_train_fold, theta_singleton, theta_pair, theta, phi_placeholder$inlier_component, phi_placeholder$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), 0.1, 1e-20, mu_init, FALSE)
				#mu_init2 <- mu_list$probability
				#mu_pairwise_init2 <- mu_list$probability_pairwise
				#lbfgs_output <- lbfgs(compute_vi_crf_likelihood_for_lbfgs, compute_vi_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat_train_fold, discrete_outliers=outliers_train_fold, posterior=outliers_train_fold, posterior_pairwise=pairwise_outliers_train_fold, phi=phi_placeholder, lambda=lambda, lambda_pair=0, lambda_singleton=0,mu_init=mu_init2, mu_pairwise_init=mu_pairwise_init2, convergence_thresh=1e-200, step_size=0.5, independent_variables=independent_variables,invisible=0)
				lbfgs_output <- grad_desc(compute_vi_crf_gradient_for_lbfgs, compute_vi_crf_likelihood_for_lbfgs, gradient_variable_vec, feat_train_fold, outliers_train_fold, outliers_train_fold, pairwise_outliers_train_fold, phi_placeholder, lambda, lambda_pair, 0, mu_init, mu_pairwise_init, 1e-3, .8, independent_variables, gradient_descent_threshold)

		}


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
			if (inference_method == "exact") {
				if (independent_variables == "false") {
					mu_list_test <- update_marginal_probabilities_exact_inference_cpp(feat_test_fold, outliers_test_fold, gam_parameters$theta_singleton, gam_parameters$theta_pair, gam_parameters$theta, phi_placeholder$inlier_component, phi_placeholder$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
					mu_test <- mu_list_test$probability
				} else if (independent_variables == "true") {
					mu_list_test <- update_independent_marginal_probabilities_exact_inference_cpp(feat_test_fold, outliers_test_fold, gam_parameters$theta_singleton, gam_parameters$theta_pair, gam_parameters$theta, phi_placeholder$inlier_component, phi_placeholder$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
					mu_test <- mu_list_test$probability
				}
			} else if (inference_method == "vi") {
				mu_init = matrix(.5, dim(feat_train_fold)[1], number_of_dimensions)
				mu_list_test <- update_marginal_probabilities_vi_cpp(feat_test_fold, outliers_test_fold, gam_parameters$theta_singleton, gam_parameters$theta_pair, gam_parameters$theta, phi_placeholder$inlier_component, phi_placeholder$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), 0.5, 1e-4, mu_init, FALSE)
				mu_test <- mu_list_test$probability
			}

			# Compute roc
			test_predictions <- as.vector(mu_test)
			test_labels <- as.vector(outliers_test_fold)
			pr_obj <- pr.curve(scores.class0 = test_predictions[test_labels==1], scores.class1 = test_predictions[test_labels==0], curve = T)
			auc <- pr_obj$auc.integral
			aucs <- c(aucs, auc)
		}
		avg_aucs <- c(avg_aucs, mean(aucs))
	}
	# Get best (one with highest avg auc across folds) lambda 
	print(avg_aucs)
	best_index <- which(avg_aucs==max(avg_aucs))[1]  # [1] for tie breakers
	best_lambda <- lambda_costs[best_index]
  best_lambda_pair <- lambda_pair_costs[best_index]
	# Using best lambda, recompute GAM
	if (inference_method == "exact") {
		lbfgs_output <- lbfgs(compute_exact_crf_likelihood_for_lbfgs, compute_exact_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat_train_shuff, discrete_outliers=binary_outliers_train_shuff, posterior=binary_outliers_train_shuff, posterior_pairwise=pairwise_binary_outliers_train_shuff, phi=phi_placeholder, lambda=best_lambda, lambda_pair=best_lambda_pair, lambda_singleton=0, independent_variables=independent_variables,invisible=1)
  } else if (inference_method == "vi") {
		mu_init = matrix(.5, dim(feat_train_fold)[1], number_of_dimensions)

		mu_list <- update_marginal_probabilities_vi_cpp(feat_train_shuff, binary_outliers_train_shuff, theta_singleton, theta_pair, theta, phi_placeholder$inlier_component, phi_placeholder$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), 0.5, 1e-4, mu_init, FALSE)
		mu_init2 <- mu_list$probability
		mu_pairwise_init2 <- mu_list$probability_pairwise

		#lbfgs_output <- lbfgs(compute_vi_crf_likelihood_for_lbfgs, compute_vi_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat_train_shuff, discrete_outliers=binary_outliers_train_shuff, posterior=binary_outliers_train_shuff, posterior_pairwise=pairwise_binary_outliers_train_shuff, phi=phi_placeholder, lambda=best_lambda, lambda_pair=0, lambda_singleton=0,mu_init=mu_init2, mu_pairwise_init=mu_pairwise_init2, convergence_thresh=1e-200, step_size=0.5, independent_variables=independent_variables,invisible=0)
		lbfgs_output <- grad_desc(compute_vi_crf_gradient_for_lbfgs, compute_vi_crf_likelihood_for_lbfgs, gradient_variable_vec, feat_train_shuff, binary_outliers_train_shuff, binary_outliers_train_shuff, pairwise_binary_outliers_train_shuff, phi_placeholder, best_lambda, best_lambda_pair, 0, mu_init2, mu_pairwise_init2, 1e-3, .8, independent_variables, gradient_descent_threshold)


	}
	# Get optimized crf coefficients back into model_params format
	gam_parameters$theta_singleton <- lbfgs_output$par[1:number_of_dimensions]
	for (dimension in 1:number_of_dimensions) {
		gam_parameters$theta[,dimension] <- lbfgs_output$par[(number_of_dimensions + 1 + ncol(feat_train)*(dimension-1)):(number_of_dimensions + ncol(feat_train)*(dimension))]
	}
 	#gam_parameters$theta_pair[1,] <- lbfgs_output$par[(number_of_dimensions + (number_of_dimensions*ncol(feat_train)) + 1):length(lbfgs_output$par)]
	gam_parameters$theta_pair <- matrix(lbfgs_output$par[(number_of_dimensions + (number_of_dimensions*ncol(feat_train)) + 1):length(lbfgs_output$par)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)
	return(list(lambda=best_lambda, lambda_pair=best_lambda_pair, gam_parameters=gam_parameters))
}



get_discretized_outliers <- function(outlier_pvalues) {
	# initialize output
	outliers_discretized <- matrix(0,dim(outlier_pvalues)[1], dim(outlier_pvalues)[2])
	for (dimension in 1:ncol(outlier_pvalues)) {
		# Check if it is total expression
		if (min(outlier_pvalues[,dimension], na.rm=TRUE) < 0) {
			under_expression = outlier_pvalues[,dimension] < 0 & !is.na(outlier_pvalues[,dimension])
			log_pvalues = -log10(abs(outlier_pvalues[,dimension]) + 1e-6)
			log_pvalues[under_expression] = log_pvalues[under_expression]*-1
			#discretized <- cut(log_pvalues,breaks=c(-6.01,-4,-2,-1,1,2,4,6.01))
			discretized <- cut(log_pvalues, breaks=c(-6.01,-1,1,6.01))
		} else {
			log_pvalues = -log10(abs(outlier_pvalues[,dimension]) + 1e-6)
			# discretized <- cut(log_pvalues, 7)
			discretized <- cut(log_pvalues, breaks=c(-.01,1,4,6))
		}
		outliers_discretized[,dimension] = as.numeric(discretized)
	}
	colnames(outliers_discretized) = colnames(outlier_pvalues)
	return(outliers_discretized)
}


load_watershed_data <- function(input_file, number_of_dimensions, pvalue_fraction) {
	raw_data <- read.table(input_file, header=TRUE)
	# Get genomic features (first 2 columns are line identifiers and last (number_of_dimensions+1) columns are outlier status' and N2 pair
	feat <- raw_data[,3:(ncol(raw_data)-number_of_dimensions-1)]
	# sample name as SubjectID:GeneName
	rownames(feat) <- paste(raw_data[,"SubjectID"], ":", raw_data[,"GeneName"],sep="")
	# Pvalues of outlier status of a particular sample (rows) for a particular outlier type (columns)
	outlier_pvalues <- raw_data[,(ncol(raw_data)-number_of_dimensions):(ncol(raw_data)-1)]
	# sample name as SubjectID:GeneName
	rownames(outlier_pvalues) <- paste(raw_data[,"SubjectID"], ":", raw_data[,"GeneName"],sep="")
	# Convert outlier status into binary random variables
	outliers_binary <- ifelse(abs(outlier_pvalues)<=.1,1,0) # Strictly for initialization of binary output matrix
	for (dimension_num in 1:number_of_dimensions) {
		ordered <- sort(abs(outlier_pvalues[,dimension_num]))
		max_val <- ordered[floor(length(ordered)*pvalue_fraction)]
		outliers_binary[,dimension_num] <- ifelse(abs(outlier_pvalues[,dimension_num])<=max_val,1,0)
	}
	# Convert outlier status into discretized random variables
	outliers_discrete <- get_discretized_outliers(outlier_pvalues)
	# Extract array of N2 pairs
	N2_pairs=factor(raw_data[,"N2pair"], levels=unique(raw_data[,"N2pair"]))
	# Put all data into compact data structure
	data_input <- list(feat=as.matrix(feat), outlier_pvalues=as.matrix(outlier_pvalues),outliers_binary=as.matrix(outliers_binary),outliers_discrete=outliers_discrete, N2_pairs=N2_pairs)
	return(data_input)
}



initialize_model_params <- function(num_samples, num_genomic_features, number_of_dimensions, phi_init, theta_pair_init, theta_singleton_init, theta_init, pseudoc, lambda, lambda_singleton, lambda_pair, inference_method, independent_variables) {

	model_params <- list(theta_pair = theta_pair_init, 
						 theta_singleton = theta_singleton_init,
						 theta = theta_init,
						 mu = matrix(.5, num_samples, number_of_dimensions),
						 mu_pairwise = matrix(.5, num_samples, choose(number_of_dimensions, 2)),
						 posterior = matrix(.5, num_samples, number_of_dimensions),
						 posterior_pairwise = matrix(.5, num_samples, choose(number_of_dimensions, 2)),
						 num_samples = num_samples,
						 num_genomic_features = num_genomic_features,
						 number_of_dimensions = number_of_dimensions,
						 phi = phi_init,
						 lambda = lambda,
						 lambda_singleton = lambda_singleton,
						 lambda_pair = lambda_pair,
						 pseudoc = pseudoc,
						 observed_data_log_likelihood = 0,
						 independent_variables = independent_variables,
						 inference_method = inference_method)

   return(model_params)
}




update_marginal_posterior_probabilities <- function(feat, discrete_outliers, model_params) {
	if (model_params$inference_method == "exact") {
		if (model_params$independent_variables == "false") {
		posterior_list <- update_marginal_probabilities_exact_inference_cpp(feat, discrete_outliers, model_params$theta_singleton, model_params$theta_pair, model_params$theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2), TRUE)
		} else if (model_params$independent_variables == "true") {
			posterior_list <- update_independent_marginal_probabilities_exact_inference_cpp(feat, discrete_outliers, model_params$theta_singleton, model_params$theta_pair, model_params$theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2), TRUE)
		}
	} else if (model_params$inference_method == "vi") {
		#posterior_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, model_params$theta_singleton, model_params$theta_pair, model_params$theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2), 0.1, 1e-100, model_params$posterior, TRUE)
		posterior_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, model_params$theta_singleton, model_params$theta_pair, model_params$theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2), 0.5, 1e-6, model_params$posterior, TRUE)

	}
	return(posterior_list)
}

# Extract gradient variable vector
# First model_params$number_of_dimensions terms are intercepts for each dimension
# Next there are model_params$number_of_dimensions chunks of length $number_of_genomic_features (each chunk is that dimension's beta)
# Next there are model_params$number_of_dimensions choose 2 theta_pairs
extract_gradient_variable_vector <- function(model_params) {
	# Initialize vector
	x <- c()
	# Add theta_singleton (intercepts)
	x <- c(x, model_params$theta_singleton)
	# Add theta for each dimension (betas)
	for (dimension in 1:model_params$number_of_dimensions) {
		x <- c(x, model_params$theta[, dimension])
	}
	# Add theta_pair (edges between unobserved nodes)
	# x <- c(x, model_params$theta_pair[1,])
	# Add theta_pair (edges between unobserved nodes)
	for (row_number in 1:(dim(model_params$theta_pair)[1])) {
		x <- c(x, model_params$theta_pair[row_number,])
	}

	return(x)
}


# Calculate gradient of crf likelihood (fxn formatted to be used in LBFGS)
compute_vi_crf_gradient_for_lbfgs <- function(x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables) {


	# Extract relevent scalers describing data
	num_genomic_features <- ncol(feat)
	num_samples <- nrow(feat)
	number_of_dimensions <- ncol(discrete_outliers)

	# Get crf coefficients back into inference format
	theta_singleton <- x[1:number_of_dimensions]
	theta <- matrix(0,num_genomic_features,number_of_dimensions)
	for (dimension in 1:number_of_dimensions) {
		theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
	}

	theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)



	# Compute expected value of the CRFs (mu)
	# mu_list <- update_marginal_probabilities_exact_inference_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
	#mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, mu_init, FALSE)
	#mu <- mu_list$probability
	#mu_pairwise <- mu_list$probability_pairwise
	mu <- mu_init
	mu_pairwise <- mu_pairwise_init

	# Gradient of singleton terms (intercepts)
	grad_singleton <- (colSums(posterior) - colSums(mu))*(1/nrow(posterior)) - lambda_singleton*theta_singleton

	# Gradient of theta terms (betas)
	theta_vec <- x[(number_of_dimensions+1):(length(x)-(choose(number_of_dimensions, 2)*nrow(theta_pair)))]
	grad_theta <- c()
	for (dimension in 1:number_of_dimensions) {
		temp_grad <- colSums(posterior[,dimension]*feat) - colSums(mu[,dimension]*feat)
		grad_theta <- c(grad_theta, temp_grad)
	}

	grad_theta <- grad_theta*(1/nrow(posterior)) - lambda*theta_vec

	# Gradient of theta pair terms (edges)
	if (independent_variables == "true") {
		grad_pair <- numeric(nrow(posterior_pairwise))
	} else if (independent_variables == "false") {
		grad_pair <- (colSums(posterior_pairwise) - colSums(mu_pairwise))*(1/nrow(posterior_pairwise)) - lambda_pair*theta_pair[1,]
	} else if (independent_variables == "false_geno") {
		for (theta_pair_dimension in 1:(dim(theta_pair)[1])) {
			if (theta_pair_dimension == 1) {
				grad_pair <- (colSums(posterior_pairwise) - colSums(mu_pairwise))*(1/nrow(posterior_pairwise)) - lambda_pair*theta_pair[theta_pair_dimension,]
			} else {
				temp_grad_pair <- (colSums(posterior_pairwise*feat[,(theta_pair_dimension-1)]) - colSums(mu_pairwise*feat[,(theta_pair_dimension-1)]))*(1/nrow(posterior_pairwise)) - lambda*theta_pair[theta_pair_dimension,]
				grad_pair <- c(grad_pair, temp_grad_pair)
			}
		}
	}

	# Merge all gradients
	grad <- c(grad_singleton, grad_theta, grad_pair)
	#print("analytical gradient")
	#print(as.vector(-grad))
	#print("Numerical gradient")
	#num_vi_grad <- grad(compute_vi_crf_likelihood_for_lbfgs, x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=0, lambda_singleton=0, mu_init=mu_init, mu_pairwise_init=mu_pairwise_init, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
	#print(num_vi_grad)
	return(-grad)
}

compute_vi_crf_likelihood_for_lbfgs <- function(x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables) {
	# Extract relevent scalers describing data
	num_genomic_features <- ncol(feat)
	num_samples <- nrow(feat)
	number_of_dimensions <- ncol(discrete_outliers)

	# Get crf coefficients back into inference format
	theta_singleton <- x[1:number_of_dimensions]
	theta <- matrix(0,num_genomic_features,number_of_dimensions)
	for (dimension in 1:number_of_dimensions) {
		theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
	}
	#theta_pair <- matrix(0,1, choose(number_of_dimensions, 2))
	#theta_pair[1,] <- x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)]
	theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

	# Compute expected value of the CRFs (mu)
	#mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, mu_init, FALSE)
	#mu <- mu_list$probability
	#mu_pairwise <- mu_list$probability_pairwise


	# Compute likelihood in cpp function
	#####################################
	#log_likelihood <- compute_crf_likelihood_vi_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton, mu, mu_pairwise)
	#####################################
	log_likelihood <- compute_crf_likelihood_vi_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init)
	#print("Analytical likelihood")
	#print(-log_likelihood)
	return(-log_likelihood)
}






# Calculate gradient of crf likelihood (fxn formatted to be used in LBFGS)
compute_exact_crf_gradient_for_lbfgs <- function(x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, independent_variables) {
	# Extract relevent scalers describing data
	num_genomic_features <- ncol(feat)
	num_samples <- nrow(feat)
	number_of_dimensions <- ncol(discrete_outliers)

	# Get crf coefficients back into inference format
	theta_singleton <- x[1:number_of_dimensions]
	theta <- matrix(0,num_genomic_features,number_of_dimensions)
	for (dimension in 1:number_of_dimensions) {
		theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
	}
	theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)



	# Compute expected value of the CRFs (mu)
	if (independent_variables == "false") {
		mu_list <- update_marginal_probabilities_exact_inference_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
		mu <- mu_list$probability
		mu_pairwise <- mu_list$probability_pairwise
	} else if (independent_variables == "true") {
		mu_list <- update_independent_marginal_probabilities_exact_inference_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
		mu <- mu_list$probability
		mu_pairwise <- mu_list$probability_pairwise
	}


	# Gradient of singleton terms (intercepts)
	grad_singleton <- (colSums(posterior) - colSums(mu))*(1/nrow(posterior)) - lambda_singleton*theta_singleton

	# Gradient of theta terms (betas)
	theta_vec <- x[(number_of_dimensions+1):(length(x)-(choose(number_of_dimensions, 2)*nrow(theta_pair)))]
	grad_theta <- c()
	for (dimension in 1:number_of_dimensions) {
		temp_grad <- colSums(posterior[,dimension]*feat) - colSums(mu[,dimension]*feat)
		grad_theta <- c(grad_theta, temp_grad)
	}

	grad_theta <- grad_theta*(1/nrow(posterior)) - lambda*theta_vec

	# Gradient of theta pair terms (edges)
	if (independent_variables == "true") {
		grad_pair <- numeric(nrow(posterior_pairwise))
	} else if (independent_variables == "false") {
		grad_pair <- (colSums(posterior_pairwise) - colSums(mu_pairwise))*(1/nrow(posterior_pairwise)) - lambda_pair*theta_pair[1,]
	} else if (independent_variables == "false_geno") {
		for (theta_pair_dimension in 1:(dim(theta_pair)[1])) {
			if (theta_pair_dimension == 1) {
				grad_pair <- (colSums(posterior_pairwise) - colSums(mu_pairwise))*(1/nrow(posterior_pairwise)) - lambda_pair*theta_pair[theta_pair_dimension,]
			} else {
				temp_grad_pair <- (colSums(posterior_pairwise*feat[,(theta_pair_dimension-1)]) - colSums(mu_pairwise*feat[,(theta_pair_dimension-1)]))*(1/nrow(posterior_pairwise)) - lambda*theta_pair[theta_pair_dimension,]
				grad_pair <- c(grad_pair, temp_grad_pair)
			}
		}
	}

	# Merge all gradients
	grad <- c(grad_singleton, grad_theta, grad_pair)
	return(-grad)
}

compute_exact_crf_likelihood_for_lbfgs <- function(x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, independent_variables) {
	# Extract relevent scalers describing data
	num_genomic_features <- ncol(feat)
	num_samples <- nrow(feat)
	number_of_dimensions <- ncol(discrete_outliers)

	# Get crf coefficients back into inference format
	theta_singleton <- x[1:number_of_dimensions]
	theta <- matrix(0,num_genomic_features,number_of_dimensions)
	for (dimension in 1:number_of_dimensions) {
		theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
	}
	#theta_pair <- matrix(0,1, choose(number_of_dimensions, 2))
	#theta_pair[1,] <- x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)]
	theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

	# Compute likelihood in cpp function
	if (independent_variables == "false") {
		log_likelihood <- compute_crf_likelihood_exact_inference_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton)
	} else if (independent_variables == "true") {
		log_likelihood <- compute_independent_crf_likelihood_exact_inference_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton)
	}
	return(-log_likelihood)
}

compute_vi_crf_likelihood_for_lbfgs2 <- function(x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, convergence_thresh, step_size, independent_variables) {
  # Extract relevent scalers describing data
  num_genomic_features <- ncol(feat)
  num_samples <- nrow(feat)
  number_of_dimensions <- ncol(discrete_outliers)

  # Get crf coefficients back into inference format
  theta_singleton <- x[1:number_of_dimensions]
  theta <- matrix(0,num_genomic_features,number_of_dimensions)
  for (dimension in 1:number_of_dimensions) {
    theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
  }
  #theta_pair <- matrix(0,1, choose(number_of_dimensions, 2))
  #theta_pair[1,] <- x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)]
  theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

  #mu_init <- readRDS(mu_init_file)

  # Compute expected value of the CRFs (mu)
  mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, global_mu, FALSE)
  mu <- mu_list$probability
  mu_pairwise <- mu_list$probability_pairwise

  global_mu <<- mu
  global_mu_pairwise <<- mu_pairwise


  # Compute likelihood in cpp function
  #####################################
  #log_likelihood <- compute_crf_likelihood_vi_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton, mu, mu_pairwise)
  #####################################
  log_likelihood <- compute_crf_likelihood_vi_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton, mu, mu_pairwise)
  #print("Analytical likelihood")
  #print(-log_likelihood)
  #saveRDS(mu, mu_init_file)
  print(-log_likelihood)
  return(-log_likelihood)
}



# Calculate gradient of crf likelihood (fxn formatted to be used in LBFGS)
compute_vi_crf_gradient_for_lbfgs2 <- function(x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, convergence_thresh, step_size, independent_variables) {
  # Extract relevent scalers describing data
  num_genomic_features <- ncol(feat)
  num_samples <- nrow(feat)
  number_of_dimensions <- ncol(discrete_outliers)

  # Get crf coefficients back into inference format
  theta_singleton <- x[1:number_of_dimensions]
  theta <- matrix(0,num_genomic_features,number_of_dimensions)
  for (dimension in 1:number_of_dimensions) {
    theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
  }

  theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)


  #mu_init <- readRDS(mu_init_file)
  # Compute expected value of the CRFs (mu)
  # mu_list <- update_marginal_probabilities_exact_inference_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), FALSE)
  #mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, global_mu_init, FALSE)
  #mu <- mu_list$probability
  #mu_pairwise <- mu_list$probability_pairwise
  #mu <- mu_init
  #mu_pairwise <- mu_pairwise_init

  mu <- global_mu
  mu_pairwise <- global_mu_pairwise

  # Gradient of singleton terms (intercepts)
  grad_singleton <- (colSums(posterior) - colSums(mu))*(1/nrow(posterior)) - lambda_singleton*theta_singleton

  # Gradient of theta terms (betas)
  theta_vec <- x[(number_of_dimensions+1):(length(x)-(choose(number_of_dimensions, 2)*nrow(theta_pair)))]
  grad_theta <- c()
  for (dimension in 1:number_of_dimensions) {
    temp_grad <- colSums(posterior[,dimension]*feat) - colSums(mu[,dimension]*feat)
    grad_theta <- c(grad_theta, temp_grad)
  }

  grad_theta <- grad_theta*(1/nrow(posterior)) - lambda*theta_vec

  # Gradient of theta pair terms (edges)
  if (independent_variables == "true") {
    grad_pair <- numeric(nrow(posterior_pairwise))
  } else if (independent_variables == "false") {
    grad_pair <- (colSums(posterior_pairwise) - colSums(mu_pairwise))*(1/nrow(posterior_pairwise)) - lambda_pair*theta_pair[1,]
  } else if (independent_variables == "false_geno") {
    for (theta_pair_dimension in 1:(dim(theta_pair)[1])) {
      if (theta_pair_dimension == 1) {
        grad_pair <- (colSums(posterior_pairwise) - colSums(mu_pairwise))*(1/nrow(posterior_pairwise)) - lambda_pair*theta_pair[theta_pair_dimension,]
      } else {
        temp_grad_pair <- (colSums(posterior_pairwise*feat[,(theta_pair_dimension-1)]) - colSums(mu_pairwise*feat[,(theta_pair_dimension-1)]))*(1/nrow(posterior_pairwise)) - lambda*theta_pair[theta_pair_dimension,]
        grad_pair <- c(grad_pair, temp_grad_pair)
      }
    }
  }

  # Merge all gradients
  grad <- c(grad_singleton, grad_theta, grad_pair)
  print("GRAD")
  #global_iter <<- global_iter + 1
  #print(global_iter)
  #if (global_iter >= 3) {
  #  global_mu_init <<- mu
  #}
  #saveRDS(mu, mu_init_file)
  #print("analytical gradient")
  #print(as.vector(-grad))
  #print("Numerical gradient")
  #num_vi_grad <- grad(compute_vi_crf_likelihood_for_lbfgs, x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=0, lambda_singleton=0, mu_init=mu_init, mu_pairwise_init=mu_pairwise_init, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
  #print(num_vi_grad)
  return(-grad)
}


# Compute MAP estimates of the coefficients defining the conditional random field (CRF)
map_crf <- function(feat, discrete_outliers, model_params, iter) {
	# Extract gradient variable vector
	# First model_params$number_of_dimensions terms are intercepts for each dimension
	# Next there are model_params$number_of_dimensions chunks of length $number_of_genomic_features (each chunk is that dimension's beta)
	# Next there are model_params$number_of_dimensions choose 2 theta_pairs
	gradient_variable_vec <- extract_gradient_variable_vector(model_params)


	# Run LBFGS (https://cran.r-project.org/web/packages/lbfgs/lbfgs.pdf) using our gradient and likelihood functions.
	if (model_params$inference_method == "exact") {
		lbfgs_output <- lbfgs(compute_exact_crf_likelihood_for_lbfgs, compute_exact_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat, discrete_outliers=discrete_outliers, posterior=model_params$posterior, posterior_pairwise=model_params$posterior_pairwise, phi=model_params$phi, lambda=model_params$lambda, lambda_pair=model_params$lambda_pair, lambda_singleton=model_params$lambda_singleton, independent_variables=model_params$independent_variables, invisible=1)
	} else if (model_params$inference_method == "vi") {
		theta_singleton <- gradient_variable_vec[1:model_params$number_of_dimensions]
		theta <- matrix(0,model_params$num_genomic_features, model_params$number_of_dimensions)
		for (dimension in 1:model_params$number_of_dimensions) {
			theta[,dimension] <- gradient_variable_vec[(model_params$number_of_dimensions + 1 + model_params$num_genomic_features*(dimension-1)):(model_params$number_of_dimensions + model_params$num_genomic_features*(dimension))]
		}
		theta_pair <- matrix(gradient_variable_vec[(model_params$number_of_dimensions + (model_params$number_of_dimensions*model_params$num_genomic_features) + 1):length(gradient_variable_vec)], ncol=choose(model_params$number_of_dimensions, 2),byrow=TRUE)

		#mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2), 0.1, 1e-100, model_params$mu, FALSE)
		mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2), 0.8, 1e-4, model_params$mu, FALSE)
		mu_init2 <- mu_list$probability
		mu_pairwise_init2 <- mu_list$probability_pairwise

		#lbfgs_output <- lbfgs(compute_vi_crf_likelihood_for_lbfgs, compute_vi_crf_gradient_for_lbfgs, gradient_variable_vec, feat=feat, discrete_outliers=discrete_outliers, posterior=model_params$posterior, posterior_pairwise=model_params$posterior_pairwise, phi=model_params$phi, lambda=model_params$lambda, lambda_pair=0, lambda_singleton=0,mu_init=mu_init2, mu_pairwise_init=mu_pairwise_init2, convergence_thresh=1e-200, step_size=0.5, independent_variables=model_params$independent_variables,invisible=0)
		#lbfgs_output <- grad_desc(compute_vi_crf_gradient_for_lbfgs, compute_vi_crf_likelihood_for_lbfgs, gradient_variable_vec, feat, discrete_outliers, model_params$posterior, model_params$posterior_pairwise, model_params$phi, model_params$lambda, model_params$lambda_pair, model_params$lambda_singleton, mu_init2, mu_pairwise_init2, 1e-3, .8, independent_variables, model_params$gradient_descent_threshold, model_params$gradient_descent_stepsize, iter)
    #global_iter <<- 0
    global_mu <<- mu_init2
    global_mu_pairwise <<- mu_pairwise_init2
    lbfgs_output <- lbfgs(compute_vi_crf_likelihood_for_lbfgs2, compute_vi_crf_gradient_for_lbfgs2, gradient_variable_vec, feat=feat, discrete_outliers=discrete_outliers, posterior=model_params$posterior, posterior_pairwise=model_params$posterior_pairwise, phi=model_params$phi, lambda=model_params$lambda, lambda_pair=model_params$lambda_pair, lambda_singleton=model_params$lambda_singleton, independent_variables=independent_variables,step_size=.3,convergence_thresh=1e-20)

	}
	# Check to make sure LBFGS converged OK
	if (lbfgs_output$convergence != 0) {
		print(paste0("LBFGS optimazation on CRF did not converge. It reported convergence error of: ", lbfgs_output$convergence))
		print(lbfgs_output$message)
	}

	# Get optimized crf coefficients back into model_params format
	model_params$theta_singleton <- lbfgs_output$par[1:model_params$number_of_dimensions]
	for (dimension in 1:model_params$number_of_dimensions) {
		model_params$theta[,dimension] <- lbfgs_output$par[(model_params$number_of_dimensions + 1 + ncol(feat)*(dimension-1)):(model_params$number_of_dimensions + ncol(feat)*(dimension))]
	}

 	model_params$theta_pair <- matrix(lbfgs_output$par[(model_params$number_of_dimensions + (model_params$number_of_dimensions*ncol(feat)) + 1):length(lbfgs_output$par)], ncol=choose(model_params$number_of_dimensions, 2),byrow=TRUE)

	return(model_params)
}

# Compute MAP estimates of the coefficients defined by P(outlier_status| FR)
map_phi <- function(discrete_outliers, model_params) {
	num_bins = 3
	# Initialize output matrices
	phi_outlier <- matrix(1,model_params$number_of_dimensions, num_bins)	
	phi_inlier <- matrix(1,model_params$number_of_dimensions, num_bins)
	# Count number of times we fall into each bin
	for (bin_number in 1:num_bins) {
    	phi_outlier[,bin_number] <- colSums(((discrete_outliers==bin_number)*model_params$posterior),na.rm=TRUE)
    	phi_inlier[,bin_number] <- colSums(((discrete_outliers==bin_number)*(1-model_params$posterior)),na.rm=TRUE)
  }

  # Add prior
  for (dimension_number in 1:model_params$number_of_dimensions) {
    phi_outlier[dimension_number,] = phi_outlier[dimension_number,] + model_params$pseudoc[dimension_number]
    phi_inlier[dimension_number,] = phi_inlier[dimension_number,] + model_params$pseudoc[dimension_number]
  }
	# Add prior
	#phi_outlier <- phi_outlier + model_params$pseudoc
	#phi_inlier <- phi_inlier + model_params$pseudoc

	# Normalize
	phi_outlier <- phi_outlier/rowSums(phi_outlier)
	phi_inlier <- phi_inlier/rowSums(phi_inlier)

	# Add to model_params
	model_params$phi$outlier_component <- phi_outlier
	model_params$phi$inlier_component <- phi_inlier
	return(model_params)
}

make_vector_to_matrix <- function(theta_pair, number_of_dimensions) {
  # Convert theta_pair vector into matrix of number_of_tissuesXnumber_of_tissues
  theta_pair_mat = matrix(0, number_of_dimensions, number_of_dimensions)
  dimension_counter = 1
  for (dimension1 in 1:number_of_dimensions) {
    for (dimension2 in dimension1:number_of_dimensions) {
      if (dimension1 != dimension2) {
        theta_pair_mat[dimension1, dimension2] = theta_pair[1, dimension_counter]
        theta_pair_mat[dimension2, dimension1] = theta_pair[1, dimension_counter]
        dimension_counter = dimension_counter + 1
      }
    }
  }
  return(theta_pair_mat)
}

integratedEM <- function(feat, discrete_outliers, phi_init, theta_pair_init, theta_singleton_init, theta_init, pseudoc, lambda, lambda_singleton, lambda_pair, number_of_dimensions, inference_method, independent_variables) {
	pseudoc_vec <- pseudoc*as.vector(colSums(!is.na(discrete_outliers)))
  print(pseudoc_vec)
  model_params <- initialize_model_params(dim(feat)[1], dim(feat)[2], number_of_dimensions, phi_init, theta_pair_init, theta_singleton_init, theta_init, pseudoc_vec, lambda, lambda_singleton, lambda_pair, inference_method, independent_variables)


	#################
	# Start loop here
	##################

  convergence = FALSE
  iter <- 1 
  while(convergence==FALSE) {

		################ E Step
		expected_posteriors <- update_marginal_posterior_probabilities(feat, discrete_outliers, model_params)

		model_params$posterior = expected_posteriors$probability
		model_params$posterior_pairwise = expected_posteriors$probability_pairwise

		################## Compute observed data log likelihood
		#observed_data_log_likelihood <- compute_exact_observed_data_log_likelihood_cpp(feat, discrete_outliers, model_params$theta_singleton, model_params$theta_pair, model_params$theta, model_params$phi$inlier_component, model_params$phi$outlier_component, model_params$number_of_dimensions, choose(model_params$number_of_dimensions, 2)) #+
		#model_params$observed_data_log_likelihood = observed_data_log_likelihood

		print('########################')
		print(model_params$theta)
		print(model_params$theta_singleton)
		print(make_vector_to_matrix(model_params$theta_pair, number_of_dimensions))
		print(model_params$phi)
		print(paste0("ITERATION ", iter))
		#print(paste0("observed data log likelihood: ", observed_data_log_likelihood))
		# saveRDS(model_params, paste0(output_root, "_model_params_iteration_",iter,".rds"))
		#  Keep track of previous iteration's parameters in order to check for convergence
		phi_old <- model_params$phi
		beta_old <- model_params$beta
		theta_singleton_old <- model_params$theta_singleton
		theta_pair_old <- model_params$theta_pair
    theta_old <- model_params$theta

		################ M Step
		# Compute MAP estimates of the coefficients defining the conditional random field (CRF)
		model_params <- map_crf(feat, discrete_outliers, model_params, iter)
		# Compute MAP estimates of the coefficients defined by P(outlier_status| FR)
		model_params <- map_phi(discrete_outliers, model_params)

    if (iter > 65 | identical(theta_old, model_params$theta)) {
      convergence=TRUE
    }
    iter <- iter + 1

	}
	return(model_params)
}







