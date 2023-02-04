args = commandArgs(trailingOnly=TRUE)
library(pROC)
library(ggplot2)
library(sigmoid)
library(Rcpp)
library(lbfgs)
library(numDeriv)
sourceCpp("crf_variational_updates.cpp")


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
  mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, global_mu_init, global_mu_init_old, global_mu_init_old2, global_mu_init_old3, global_mu_init_old4, FALSE)
  mu <- mu_list$probability
  mu_pairwise <- mu_list$probability_pairwise

  # Compute likelihood in cpp function
  #####################################
  #log_likelihood <- compute_crf_likelihood_vi_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton, mu, mu_pairwise)
  #####################################
  log_likelihood <- compute_crf_likelihood_vi_cpp(posterior, posterior_pairwise, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, lambda, lambda_pair, lambda_singleton, mu, mu_pairwise)
  #print("Analytical likelihood")
  #print(-log_likelihood)
  #saveRDS(mu, mu_init_file)

  global_mu_init_old4 <<- global_mu_init_old3
  global_mu_init_old3 <<- global_mu_init_old2
  global_mu_init_old2 <<- global_mu_init_old
  global_mu_init_old <<- global_mu_init
  global_mu_init <<- mu 
  global_mu_pairwise_init <<- mu_pairwise

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

  #mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, global_mu_init, FALSE)
  #mu <- mu_list$probability
  #mu_pairwise <- mu_list$probability_pairwise
  mu <- global_mu_init
  mu_pairwise <- global_mu_pairwise_init
  #mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, global_mu_init, FALSE)
  #mu <- mu_list$probability
  #mu_pairwise <- mu_list$probability_pairwise

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
  #global_mu_init <<- matrix(.5, nrow(global_mu_init), ncol(global_mu_init))
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

grad_desc=function(grad_fxn, log_likelihood_fxn, x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables, convergence_criteria, master_stepsize, iter) {



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

    #saveRDS(x, "x_vec.rds")
    #saveRDS(mu, "mu.rds")
    #saveRDS(mu_pairwise, "mu_pairwise.rds")

    x = x-master_stepsize*g

  number_of_dimensions <- dim(discrete_outliers)[2]
  num_genomic_features <- dim(feat)[2]
  # Get crf coefficients back into inference format
  theta_singleton_g <- g[1:number_of_dimensions]
  theta_g <- matrix(0,num_genomic_features,number_of_dimensions)
  for (dimension in 1:number_of_dimensions) {
    theta_g[,dimension] <- g[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
  }
  theta_pair_g <- matrix(g[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

    #print(head(theta_singleton_g))
    #print(head(theta_g))
    #print(head(make_vector_to_matrix(theta_pair_g, 49)))

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

    #print(head(theta_singleton))
    #print(head(theta))
    #print(head(make_vector_to_matrix(theta_pair, 49)))

    #print(head(posterior-mu))


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

gtex_v8_figure_theme <- function() {
  return(theme(plot.title = element_text(face="plain",size=8), text = element_text(size=8),axis.text=element_text(size=7), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=7), legend.title = element_text(size=8)))
}


grad_descent_viz=function(grad_fxn, log_likelihood_fxn, x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables, convergence_criteria, master_stepsize, iter, output_root) {
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

  number_of_dimensions <- dim(discrete_outliers)[2]
  num_genomic_features <- dim(feat)[2]
  convergence = FALSE
  iterations = 1
  progress=list()
  prev_likelihood = log_likelihood_fxn(x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, mu_init=mu, mu_pairwise_init=mu_pairwise, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
  print(prev_likelihood)
  x_init = x
  for (iterations in 1:7) {
    visualization_output_file <- paste0(output_root, "_grad_descent_iteration_", iterations, "_likelihood_along_gradient.pdf")
    g <- grad_fxn(x, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, mu_init=mu, mu_pairwise_init=mu_pairwise, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)

    likelihoods <- c()
    positions <- c()
    num_samples <- 15

    for (gradient_step_size in seq(-1,1, .5)) {
      print(gradient_step_size)
      x_temp = x-gradient_step_size*g

      # Get crf coefficients back into inference format
      theta_singleton <- x_temp[1:number_of_dimensions]
      theta <- matrix(0,num_genomic_features,number_of_dimensions)
      for (dimension in 1:number_of_dimensions) {
        theta[,dimension] <- x_temp[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
      }
      theta_pair <- matrix(x_temp[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x_temp)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)
      # Inference
      for (sample_num in 1:num_samples) {
        mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), step_size, convergence_thresh, mu, FALSE)
        mu <- mu_list$probability
        mu_pairwise <- mu_list$probability_pairwise

        likelihood <- log_likelihood_fxn(x_temp, feat=feat, discrete_outliers=discrete_outliers, posterior=posterior, posterior_pairwise=posterior_pairwise, phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, mu_init=mu, mu_pairwise_init=mu_pairwise, convergence_thresh=convergence_thresh, step_size=step_size, independent_variables=independent_variables)
        print(likelihood)
        likelihoods <- c(likelihoods, likelihood)
        positions <- c(positions, gradient_step_size)
      }

    }

    df <- data.frame(likelihood=likelihoods, gradient_step=positions)
    print(df)

    p <- ggplot(df, aes(x=factor(gradient_step), y=likelihood)) + 
      geom_dotplot(binaxis='y', stackdir='center',dotsize=.45,fill="slateblue2") + gtex_v8_figure_theme() +
       stat_summary(fun.data="mean_sdl", fun.args = list(mult=1), geom="crossbar", width=0.5)+
       labs(x = "Gradient Stepsize", y = "-(log likelihood)", title=paste0("EM iteration ", iter, " / gradient descent iteration ", iterations)) 

    ggsave(p, filename=visualization_output_file, width=7.2, height=5, units="in")
    #saveRDS(x, "x_vec.rds")
    #saveRDS(mu, "mu.rds")
    #saveRDS(mu_pairwise, "mu_pairwise.rds")

    x = x-master_stepsize*g


    #prev_likelihood = likelihood
    iterations = iterations + 1
  }


}

step_size = as.numeric(args[1])
convergence_thresh = as.numeric(args[2])

print(step_size)
print(convergence_thresh)

data_dir <- "/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/unsupervised_modeling/watershed_tbt_debug/"
itera <- 1
convergence_criteria <- .0001
lambda <- .01
lambda_pair <- .0001
lambda_singleton <- 0
master_stepsize <- 1
#convergence_thresh <- 1e-3
#step_size <- .6
independent_variables <- "false"
num_samples <- 3000
##############
# Load in data

discrete_outliers <- readRDS(paste0(data_dir, "itera_", itera, "_discrete_outliers.rds"))
feat <- readRDS(paste0(data_dir, "itera_", itera, "_feat.rds"))
mu_init<- readRDS(paste0(data_dir, "itera_", itera, "_mu_init.rds"))
mu_pairwise_init <- readRDS(paste0(data_dir,"itera_", itera, "_mu_pairwise_temp.rds"))
phi <- readRDS(paste0(data_dir, "itera_", itera, "_phi.rds"))
posterior <- readRDS(paste0(data_dir, "itera_", itera, "_posterior.rds"))
posterior_pairwise <- readRDS(paste0(data_dir, "itera_", itera, "_posterior_pairwise.rds"))
x <- readRDS(paste0(data_dir, "itera_", itera, "_x.rds"))
#x <- readRDS("/home-1/bstrobe1@jhu.edu/scratch/gtex_v8/rare_var/gtex_v8_rare_splice/unsupervised_modeling_temp/x_vec.rds")
set.seed(5)

#################################################
# Initialize
#################################################
number_of_dimensions <- dim(discrete_outliers)[2]
num_genomic_features <- dim(feat)[2]
# Get crf coefficients back into inference format
theta_singleton <- x[1:number_of_dimensions]
theta <- matrix(0,num_genomic_features,number_of_dimensions)
for (dimension in 1:number_of_dimensions) {
  theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
}
theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

#mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), .8, 1e-0, mu_init2, FALSE)
#mu_init <- mu_list$probability
#mu_pairwise_init <- mu_list$probability_pairwise



#################################################
row_samples <- sample(nrow(feat), num_samples)

#feat <- feat[row_samples,]
#discrete_outliers <- discrete_outliers[row_samples,]
#posterior <- posterior[row_samples,]
#posterior_pairwise <- posterior_pairwise[row_samples,]
#mu_init <- mu_init[row_samples,]
#mu_pairwise_init <- mu_pairwise_init[row_samples,]


output_root <- paste0(data_dir, "EM_iteration_", itera, "_convergence_thresh_", convergence_thresh, "_step_size_", step_size)
#grad_descent_viz(compute_vi_crf_gradient_for_lbfgs, compute_vi_crf_likelihood_for_lbfgs, x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables, convergence_criteria, master_stepsize, itera, output_root)

global_mu_init <<- mu_init
global_mu_init_old <<- mu_init
global_mu_init_old2 <<- mu_init
global_mu_init_old3 <<- mu_init
global_mu_init_old4 <<- mu_init
global_mu_pairwise_init <<- mu_pairwise_init
print("START")
lbfgs_output <- lbfgs(compute_vi_crf_likelihood_for_lbfgs2, compute_vi_crf_gradient_for_lbfgs2, x, feat=feat[row_samples,], discrete_outliers=discrete_outliers[row_samples,], posterior=posterior[row_samples,], posterior_pairwise=posterior_pairwise[row_samples,], phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, independent_variables=independent_variables,step_size=step_size,convergence_thresh=convergence_thresh)



###################################

data_dir <- "/work-zfs/abattle4/bstrober/rare_variant/gtex_v8/splicing/unsupervised_modeling/watershed_tbt_debug/"
itera <- 2
convergence_criteria <- .0001
lambda <- .01
lambda_pair <- .01
lambda_singleton <- 0
master_stepsize <- 1
#convergence_thresh <- 1e-3
#step_size <- .6
independent_variables <- "false"
num_samples <- 3000
##############
# Load in data

discrete_outliers <- readRDS(paste0(data_dir, "itera_", itera, "_discrete_outliers.rds"))
feat <- readRDS(paste0(data_dir, "itera_", itera, "_feat.rds"))
mu_init2 <- readRDS(paste0(data_dir, "itera_", itera, "_mu_init.rds"))
mu_pairwise_init2 <- readRDS(paste0(data_dir,"itera_", itera, "_mu_pairwise_temp.rds"))
phi <- readRDS(paste0(data_dir, "itera_", itera, "_phi.rds"))
posterior <- readRDS(paste0(data_dir, "itera_", itera, "_posterior.rds"))
posterior_pairwise <- readRDS(paste0(data_dir, "itera_", itera, "_posterior_pairwise.rds"))
x <- readRDS(paste0(data_dir, "itera_", itera, "_x.rds"))
#x <- readRDS("/home-1/bstrobe1@jhu.edu/scratch/gtex_v8/rare_var/gtex_v8_rare_splice/unsupervised_modeling_temp/x_vec.rds")


#################################################
# Initialize
#################################################
number_of_dimensions <- dim(discrete_outliers)[2]
num_genomic_features <- dim(feat)[2]
# Get crf coefficients back into inference format
theta_singleton <- x[1:number_of_dimensions]
theta <- matrix(0,num_genomic_features,number_of_dimensions)
for (dimension in 1:number_of_dimensions) {
  theta[,dimension] <- x[(number_of_dimensions + 1 + num_genomic_features*(dimension-1)):(number_of_dimensions + num_genomic_features*(dimension))]
}
theta_pair <- matrix(x[(number_of_dimensions + (number_of_dimensions*num_genomic_features) + 1):length(x)], ncol=choose(number_of_dimensions, 2),byrow=TRUE)

#mu_list <- update_marginal_probabilities_vi_cpp(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi$inlier_component, phi$outlier_component, number_of_dimensions, choose(number_of_dimensions, 2), .8, 1e-0, mu_init2, FALSE)
#mu_init <- mu_list$probability
#mu_pairwise_init <- mu_list$probability_pairwise



#################################################
#row_samples <- sample(nrow(feat), num_samples)

#feat <- feat[row_samples,]
#discrete_outliers <- discrete_outliers[row_samples,]
#posterior <- posterior[row_samples,]
#posterior_pairwise <- posterior_pairwise[row_samples,]
#mu_init <- mu_init[row_samples,]
#mu_pairwise_init <- mu_pairwise_init[row_samples,]


#output_root <- paste0(data_dir, "EM_iteration_", itera, "_convergence_thresh_", convergence_thresh, "_step_size_", step_size)
#grad_descent_viz(compute_vi_crf_gradient_for_lbfgs, compute_vi_crf_likelihood_for_lbfgs, x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables, convergence_criteria, master_stepsize, itera, output_root)



#lbfgs_output <- lbfgs(compute_vi_crf_likelihood_for_lbfgs2, compute_vi_crf_gradient_for_lbfgs2, x, feat=feat[row_samples,], discrete_outliers=discrete_outliers[row_samples,], posterior=posterior[row_samples,], posterior_pairwise=posterior_pairwise[row_samples,], phi=phi, lambda=lambda, lambda_pair=lambda_pair, lambda_singleton=lambda_singleton, independent_variables=independent_variables,step_size=step_size,convergence_thresh=convergence_thresh)

#feat <- feat[row_samples,]
#discrete_outliers <- discrete_outliers[row_samples,]
#posterior <- posterior[row_samples,]
#posterior_pairwise <- posterior_pairwise[row_samples,]
#mu_init <- mu_init[row_samples,]
#mu_pairwise_init <- mu_pairwise_init[row_samples,]



#obj <- grad_desc(compute_vi_crf_gradient_for_lbfgs, compute_vi_crf_likelihood_for_lbfgs, x, feat, discrete_outliers, posterior, posterior_pairwise, phi, lambda, lambda_pair, lambda_singleton, mu_init, mu_pairwise_init, convergence_thresh, step_size, independent_variables, convergence_criteria, master_stepsize, itera)

