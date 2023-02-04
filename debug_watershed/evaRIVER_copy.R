args = commandArgs(trailingOnly=TRUE)
library(glmnet)
library(methods)
library(stats)
library(pROC)
library(Biobase)
library(ggplot2)



#' Posterior probabilities of FR given G and E.
#'
#' \code{getFuncRvPosteriors} computes posterior probabilities of functionality
#'         of regulatory variant (FR) given genomic features (G) and outlier
#'         status (E) with current estimate of beta (parameters between FR and G)
#'         and theta (parameters between FR and E).
#'
#' @param Out Binary values of outlier status (E).
#' @param probFuncRvFeat probabilities of FR given genomic features and estimated
#'         beta, P(FR | G, beta), from \code{getFuncRvFeat}.
#' @param theta Current estimate of theta.
#'
#' @return posterior probabilities of FR (P(FR | G, E, beta, theta)) and probable
#'         status of FR.
#'
#' @author Yungil Kim, \email{ipw012@@gmail.com}
#'
#' @examples
#' dataInput <- getData(filename=system.file("extdata", "simulation_RIVER.gz",
#'         package = "RIVER"), ZscoreThrd=1.5)
#' Feat <- scale(t(Biobase::exprs(dataInput))) # genomic features (G)
#' Out <- as.vector(as.numeric(unlist(dataInput$Outlier))-1) # outlier status (E)
#' theta.init<-matrix(c(.99, .01, .3, .7), nrow=2)
#' costs<-c(100, 10, 1, .1, .01, 1e-3, 1e-4)
#' logisticAllCV <- glmnet::cv.glmnet(Feat, Out, lambda=costs, family="binomial",
#'         alpha=0, nfolds=10)
#' probFuncRvFeat <- getFuncRvFeat(Feat, logisticAllCV$glmnet.fit,
#'         logisticAllCV$lambda.min)
#' posteriors <- getFuncRvPosteriors(Out, probFuncRvFeat, theta=theta.init)
#'
#' @export

getFuncRvPosteriors <- function(Out, probFuncRvFeat, theta) {
  probOut_FuncRv <- matrix(NA,length(Out), 2)
  probOut <- matrix(NA, length(Out), 1)
  probOut_FuncRv <- theta[Out+1,]
  probOut <-
    rowSums(probOut_FuncRv*cbind(1.0-probFuncRvFeat,probFuncRvFeat))
  post <-
    probOut_FuncRv*c(1-probFuncRvFeat,probFuncRvFeat)/c(probOut, probOut)
  list(posterior=post, mle = max.col(post)-1)
}

#' Maximum likelihoood estimate of theta.
#'
#' \code{mleTheta} computes maximum likelihoood estimate of theta (parameters
#'         between FR (functionality of regulatory variant) and E (outlier
#'         status); Naive-Bayes).
#'
#' @param Out Binary values of outlier status (E).
#' @param FuncRv Soft-assignments of FR from E-step
#' @param pseudocount Pseudo count.
#'
#' @return MLE of theta
#'
#' @author Yungil Kim, \email{ipw012@@gmail.com}
#'
#' @examples
#' dataInput <- getData(filename=system.file("extdata", "simulation_RIVER.gz",
#'         package = "RIVER"), ZscoreThrd=1.5)
#' Feat <- scale(t(Biobase::exprs(dataInput))) # genomic features (G)
#' Out <- as.vector(as.numeric(unlist(dataInput$Outlier))-1) # outlier status (E)
#' theta.init <- matrix(c(.99, .01, .3, .7), nrow=2)
#' costs <- c(100, 10, 1, .1, .01, 1e-3, 1e-4)
#' logisticAllCV <- glmnet::cv.glmnet(Feat, Out, lambda=costs, family="binomial",
#'         alpha = 0, nfolds=10)
#' probFuncRvFeat <- getFuncRvFeat(Feat, logisticAllCV$glmnet.fit, logisticAllCV$lambda.min)
#' posteriors <- getFuncRvPosteriors(Out, probFuncRvFeat, theta=theta.init)
#' thetaCur <- mleTheta(Out, FuncRv=posteriors$posterior, pseudoc=50)
#'
#' @export

mleTheta <- function(Out, FuncRv, pseudocount) {
  countOut <- matrix(NA, 2, 2)
  countOut[1,1] <- sum((Out==0)*FuncRv[,1])
  countOut[1,2] <- sum((Out==0)*FuncRv[,2])
  countOut[2,1] <- sum((Out==1)*FuncRv[,1])
  countOut[2,2] <- sum((Out==1)*FuncRv[,2])
  countOut <- countOut + pseudocount
  return(countOut/rbind(colSums(countOut), colSums(countOut)))
}

#' Maximum likelihoood estimate of beta.
#'
#' \code{mleBeta} computes maximum likelihoood estimate of beta (parameters
#'         between FR (functionality of regulatory variant) and G (genomic
#'         annotations); multivariate logistic regression).
#'
#' @param Feat Genomic features (G)
#' @param FuncRv Soft-assignments of FR from E-step
#' @param costs Candidate penalty parameter values for L2-regularization within
#'         logistic regression.
#'
#' @return MLE of beta
#'
#' @section Warning: To input a vector of candidate penalty values makes
#'         \code{glmnet} faster than to input a single penalty value
#'
#' @author Yungil Kim, \email{ipw012@@gmail.com}
#'
#' @examples
#' dataInput <- getData(filename=system.file("extdata", "simulation_RIVER.gz",
#'         package = "RIVER"), ZscoreThrd=1.5)
#' Feat <- scale(t(Biobase::exprs(dataInput))) # genomic features (G)
#' Out <- as.vector(as.numeric(unlist(dataInput$Outlier))-1) # outlier status (E)
#' theta.init <- matrix(c(.99, .01, .3, .7), nrow=2)
#' costs <- c(100, 10, 1, .1, .01, 1e-3, 1e-4)
#' logisticAllCV <- glmnet::cv.glmnet(Feat, Out, lambda=costs, family="binomial",
#'         alpha=0, nfolds=10)
#' probFuncRvFeat <- getFuncRvFeat(Feat, logisticAllCV$glmnet.fit, logisticAllCV$lambda.min)
#' posteriors <- getFuncRvPosteriors(Out, probFuncRvFeat, theta=theta.init)
#' logistic.cur <- mleBeta(Feat, FuncRv=posteriors$posterior, costs)
#'
#' @seealso \code{\link[glmnet]{glmnet}}
#'
#' @export

mleBeta <- function(Feat, FuncRv, costs) {
  glmnet(Feat, FuncRv, lambda=costs, family="binomial", alpha = 0)
}

#' Posterior probabilities of FR given G
#'
#' \code{getFuncRvFeat} computes posterior probabilities of FR (functionality of
#'         regulatory variant) given G (genomic features) and current estimate
#'         of beta (parameters between FR and G).
#'
#' @param Feat Genomic features (G)
#' @param logistic.model Logistic regression model with current estimate of beta
#' @param lambda Selected lambda
#'
#' @return probabilities of FR given genomic features, P(FR | G)
#'
#' @author Yungil Kim, \email{ipw012@@gmail.com}
#'
#' @examples
#' dataInput <- getData(filename=system.file("extdata", "simulation_RIVER.gz",
#'         package = "RIVER"), ZscoreThrd=1.5)
#' Feat <- scale(t(Biobase::exprs(dataInput))) # genomic features (G)
#' Out <- as.vector(as.numeric(unlist(dataInput$Outlier))-1) # outlier status (E)
#' costs <- c(100, 10, 1, .1, .01, 1e-3, 1e-4)
#' logisticAllCV <- glmnet::cv.glmnet(Feat, Out, lambda=costs, family="binomial",
#'         alpha = 0, nfolds=10)
#' probFuncRvFeat <- getFuncRvFeat(Feat, logistic.model=logisticAllCV$glmnet.fit,
#'         lambda=logisticAllCV$lambda.min)
#'
#' @seealso \code{\link{predict}}
#'
#' @export

getFuncRvFeat <- function(Feat, logistic.model, lambda) {
  predict(logistic.model, Feat, s=lambda, type="response")
}

#' Test posterior probabilities of FR given G and E
#'
#' \code{testPosteriors} computes posterior probabilities of FR (functionality
#'         of regulatory variant) given G (genomic annotations) and E (outlier
#'         status) with estimate of beta (parameters between FR and G) and
#'         theta (parameters between FR and E).
#'
#' @param Feat Genomic features (G)
#' @param Out Binary values of outlier status (E).
#' @param emModel Estimated parameters including beta and theta via EM and
#'         selected lambdas
#'
#' @return test posterior probabilities of FR given new outlier status (E)
#'         and genomic features (G), P(FR | G, E, beta, theta), and probable
#'         status of FR.
#'
#' @author Yungil Kim, \email{ipw012@@gmail.com}
#'
#' @examples
#' dataInput <- getData(filename=system.file("extdata", "simulation_RIVER.gz",
#'         package = "RIVER"), ZscoreThrd=1.5)
#' Feat <- scale(t(Biobase::exprs(dataInput))) # genomic features (G)
#' Out <- as.vector(as.numeric(unlist(dataInput$Outlier))-1) # outlier status (E)
#' theta.init <- matrix(c(.99, .01, .3, .7), nrow=2)
#' costs <- c(100, 10, 1, .1, .01, 1e-3, 1e-4)
#' logisticAllCV <- glmnet::cv.glmnet(Feat, Out, lambda=costs, family="binomial",
#'         alpha = 0, nfolds=10)
#' emModelAll <- integratedEM(Feat, Out, logisticAllCV$lambda.min, logisticAllCV$glmnet.fit,
#'         pseudoc=50, theta.init, costs, verbose=FALSE)
#' trainedpost <- testPosteriors(Feat, Out, emModel=emModelAll)
#'
#' @seealso \code{\link{getFuncRvFeat}} and \code{\link{getFuncRvPosteriors}}
#'
#' @export

testPosteriors <- function(Feat, Out, emModel) {
  probFuncRvFeat <- getFuncRvFeat(Feat, emModel$logistic.model, emModel$lambda)
  getFuncRvPosteriors(Out, probFuncRvFeat, emModel$theta)
}


compute_log_probability <- function(Feat, Out, logistic, beta, theta, lambda, pseudocount) {
    probFuncRvFeat <- getFuncRvFeat(Feat, logistic, lambda)
    probOut_FuncRv <- matrix(NA,length(Out), 2)
    probOut <- matrix(NA, length(Out), 1)
    probOut_FuncRv <- theta[Out+1,]
    probOut <- log(rowSums(probOut_FuncRv*cbind(1.0-probFuncRvFeat,probFuncRvFeat)))
    regularizer_term <- 0
    for (j in 1:length(beta)) {
        regularizer_term <- regularizer_term + dnorm(beta[j],mean=0,sd=sqrt((1.0/lambda)),log=TRUE)
    }
    log_prob <- sum(probOut) + regularizer_term + dbeta(theta[1,1],pseudocount,pseudocount,log=TRUE) + dbeta(theta[1,2],pseudocount,pseudocount,log=TRUE)
    return(log_prob)
}


#' An iterative expectation-maximization algorithm for RIVER
#'
#' \code{integratedEM} iteratively executes e-step and m-step until it
#'         converges. This is a main function of RIVER.
#'
#' @param Feat Genomic features (G).
#' @param Out Binary values of outlier status (E).
#' @param lambda Selected lambda.
#' @param logistic.init Smart initialization of beta (parameters between
#'         FR and G) from estimate of beta with E via multivariate logistic
#'         regression.
#' @param pseudoc Pseudo count.
#' @param theta.init Initial theta (parameters between FR (functionality
#'         of regulatory variant) and E).
#' @param costs Candidate penalty parameter values for L2-regularization
#'         within logistic regression.
#' @param verbose Logical option for showing extra information on progress.
#'
#' @return Best estimate of beta and theta, final multivariate logistic
#'         regression model, and posterior probabilities of FR.
#'
#' @author Yungil Kim, \email{ipw012@@gmail.com}
#'
#' @seealso \code{\link{getFuncRvFeat}}, \code{\link{getFuncRvPosteriors}},
#'         \code{\link{mleTheta}}, \code{\link{mleBeta}},
#'         \code{\link[glmnet]{cv.glmnet}},
#'         and \url{https://github.com/ipw012/RIVER}
#'
#' @examples
#' dataInput <- getData(filename=system.file("extdata", "simulation_RIVER.gz",
#'         package = "RIVER"), ZscoreThrd=1.5)
#' Feat <- scale(t(Biobase::exprs(dataInput))) # genomic features (G)
#' Out <- as.vector(as.numeric(unlist(dataInput$Outlier))-1) # outlier status (E)
#' theta.init=matrix(c(.99, .01, .3, .7), nrow=2)
#' costs <- c(100, 10, 1, .1, .01, 1e-3, 1e-4)
#' logisticAllCV <- glmnet::cv.glmnet(Feat, Out, lambda=costs, family="binomial",
#'         alpha = 0, nfolds=10)
#' emModelAll <- integratedEM(Feat, Out, lambda=logisticAllCV$lambda.min,
#'         logistic.init=logisticAllCV$glmnet.fit, pseudoc=50, theta=theta.init,
#'         costs, verbose=FALSE)
#'
#' @export

integratedEM <- function(Feat, Out, lambda, logistic.init,
                         pseudoc, theta.init, costs,
                         verbose=FALSE){
  theta.cur <- theta.init
  beta.cur <- logistic.init$beta[,which(logistic.init$lambda == lambda)]
  logistic.cur <- logistic.init

  steps <- 1
  maxIter <- 1000  
  converged <- 0
  for (iter in 1:maxIter) {
    if (verbose) {
      cat(' *** RIVER: EM step ',steps,'\n',sep="")
    }

    ## E-step:
    ## Compute expected posterior probabilities
    ##           given current parameters and data
    probFuncRvFeat <- getFuncRvFeat(Feat, logistic.cur, lambda)
    posteriors <- getFuncRvPosteriors(Out, probFuncRvFeat, theta.cur)
    if (verbose) {
      cat('     E-step: Top 10 % Threshold of expected P(FR=1 | G, E): ',
          round(quantile(posteriors$posterior[,2], .9),4),'\n',sep='')
    }
    ## M-step:
    ## Update theta and beta
    theta.old <- theta.cur
    # ML estimate of theta
    theta.cur <- mleTheta(Out, posteriors$posterior, pseudoc)

    beta.old <- beta.cur
    # ML estimate of beta
    logistic.cur <- mleBeta(Feat, posteriors$posterior, costs)
    beta.cur <- logistic.cur$beta[,which(logistic.cur$lambda == lambda)]
    log_prob <- compute_log_probability(Feat, Out, logistic.cur, beta.cur, theta.cur, lambda, pseudoc)
    cat('    Current log probability: ', log_prob,'\n',sep='')

    if (verbose) {
      cat('     M-step: norm(theta difference) = ',
          round(norm(matrix(theta.cur)-matrix(theta.old)),4),
          ', norm(beta difference) = ',
          round(norm(matrix(beta.cur)-matrix(beta.old)),4),
          " *** \n\n", sep="")
    }

    ## Check convergence
    if ((norm(matrix(beta.cur) - matrix(beta.old)) < 1e-3) &
        (norm(theta.cur - theta.old) < 1e-3)) {
      converged <- 1
      break
    }
    steps <- steps + 1
  }

  if (converged == 1) {
    cat(" ::: EM iteration is terminated since it converges within a
        predefined tolerance (0.001) ::: \n\n\n",sep="")
  } else if ((converged == 0) && (iter == maxIter)) {
    cat(" ::: EM iteration is terminated since it reaches a
        predefined maximum value (1000) ::: \n\n\n",sep="")
  }

  list(logistic.model=logistic.cur, beta=beta.cur, theta=theta.cur,
       posteriors=posteriors, lambda=lambda)
}






load_data <- function(input_file, pvalue_threshold=.05) {
    expData <- read.table(input_file, header=TRUE)
    Feat <- expData[,3:(ncol(expData)-2)] # genomic features
    # sample name as SubjectID:GeneName
    rownames(Feat) <- paste(expData[,"SubjectID"], ":",
    expData[,"GeneName"],sep="")
    Feat <- as.matrix(t(Feat)) # feature x sample
    # outlier status, N2 pairs
    pData <-
        data.frame(Outlier=factor(ifelse((expData[,"pvalue"])<=pvalue_threshold,1,0),
        levels=c(0,1)),
        N2pair=factor(expData[,"N2pair"],
        levels=unique(expData[,"N2pair"])))
    rownames(pData) <-
        paste(expData[,"SubjectID"],":",expData[,"GeneName"],sep="")

    # descrition of outlier status and N2 pairs
    metadata <-
        data.frame(labelDescription=c("Outlier status based on pvalues",
                                  "Pairs of samples having same rare variants"),
               row.names=c("Outlier","N2pair"))
        phenoData <- new("AnnotatedDataFrame", data=pData, varMetadata=metadata)
    dataInput <- ExpressionSet(assayData=Feat, phenoData=phenoData)
    return(dataInput)
}



plot_roc <- function(evaROC, pvalue_threshold, figure_file_name) {
  pdf(figure_file_name)
  par(mar=c(6.1, 6.1, 4.1, 4.1))
  plot(NULL, xlim=c(0,1), ylim=c(0,1), 
     xlab="False positive rate", ylab="True positive rate", 
     cex.axis=1.3, cex.lab=1.6)
  abline(0, 1, col="gray")
  lines(1-evaROC$RIVER_spec, evaROC$RIVER_sens, 
      type="s", col='dodgerblue', lwd=2)
  lines(1-evaROC$GAM_spec, evaROC$GAM_sens, 
      type="s", col='mediumpurple', lwd=2)
  legend(0.7,0.2,c("RIVER","GAM"), lty=c(1,1), lwd=c(2,2),
       col=c("dodgerblue","mediumpurple"), cex=1.2, 
       pt.cex=1.2, bty="n")
  title(main=paste("Threshold = ", pvalue_threshold," / AUC: RIVER = ", round(evaROC$RIVER_auc,3), 
                 ", GAM = ", round(evaROC$GAM_auc,3),sep=""))

  dev.off()
}

f

plot_single_roc <- function(roc_object, pvalue_threshold, figure_file_name) {
  df <- data.frame(x=1 - roc_object$specificities, y=roc_object$sensitivities)
  plotter <- ggplot(data=df, aes(x=x, y=y)) + geom_line()+ geom_point() +
  labs(x="False positive rate", y="True positive rate")
  ggsave(figure_file_name, plotter)
}



plot_betas <- function(beta, pvalue_threshold, beta_figure_file_name) {
	sorted_beta <- sort(abs(beta), decreasing = TRUE)
	sorted_labels <- labels(sort(abs(beta), decreasing=TRUE))
	sorted_labels[sorted_labels=="phred"] = "cadd_phred"

	df <- data.frame(effect_size=sorted_beta,feature=factor(sorted_labels, levels=sorted_labels))

	bar_plot <- ggplot(data=df, aes(x=feature, y=effect_size)) + geom_bar(stat="identity") +
	theme(text = element_text(size=10),axis.text=element_text(size=9), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=9), legend.title = element_text(size=10), axis.text.x = element_text(angle = 90, hjust = 1)) +
	labs(x = "Feature", y = "abs(effect size)", title=paste0("p=",pvalue_threshold))


	ggsave(bar_plot, file=beta_figure_file_name,width = 16,height=10.5,units="cm")

}







########################
# Command Line Args
########################
pvalue_threshold = as.numeric(args[1])  #Threshold for outlier calling
input_file = args[2]  # Input file
stem = args[3]  # Output file stem
river_run_dir = args[4]  # output Dir

pseudoc=50
theta_init=matrix(c(.99, .01, .3, .7), nrow=2)
costs=c(100, 10, 1, .1, .01, 1e-3, 1e-4)
verbose=TRUE

## Extract required data
dataInput <- load_data(input_file, pvalue_threshold)

# all genomic features (G)
FeatAll <- t(exprs(dataInput))
# all outlier status (E)
OutAll <- as.numeric(unlist(dataInput$Outlier))-1
# G for training models
FeatTrng <- t(exprs(dataInput[,is.na(dataInput$N2pair)]))
# E for training models
OutTrng <- as.numeric(unlist(dataInput$Outlier[is.na(dataInput$N2pair)]))-1
# G for test
FeatTest <-
  t(cbind(exprs(dataInput[,!is.na(dataInput$N2pair)])
  [,seq(from=1,to=sum(!is.na(dataInput$N2pair)),by=2)],
  exprs(dataInput[,!is.na(dataInput$N2pair)])
  [,seq(from=2,to=sum(!is.na(dataInput$N2pair)),by=2)]))

# E for test (1st and then 2nd individuals from N2 pairs)
OutTest1 <-
  as.numeric(unlist(
    c(dataInput$Outlier[!is.na(dataInput$N2pair)]
    [seq(from=1,to=sum(!is.na(dataInput$N2pair)),by=2)],
    dataInput$Outlier[!is.na(dataInput$N2pair)]
    [seq(from=2,to=sum(!is.na(dataInput$N2pair)),by=2)])))-1

# E for test (2nd and then 1st individuals from N2 pairs)
OutTest2 <-
  as.numeric(unlist(
    c(dataInput$Outlier[!is.na(dataInput$N2pair)]
    [seq(from=2,to=sum(!is.na(dataInput$N2pair)),by=2)],
    dataInput$Outlier[!is.na(dataInput$N2pair)]
    [seq(from=1,to=sum(!is.na(dataInput$N2pair)),by=2)])))-1


## Standardization
meanFeat <- apply(FeatAll, 2, mean)
sdFeat <- apply(FeatAll,2,sd)
FeatAll <- scale(FeatAll, center=meanFeat, scale=sdFeat)
FeatTrng <- scale(FeatTrng, center=meanFeat, scale=sdFeat)
# ## Generate G data for test data (Revised)
FeatTest <- scale(FeatTest, center=meanFeat, scale=sdFeat)

## Search a best lambda from a multivariate logistic regression
##         with outlier status with 10 cross-validation
## GAM (genomeic annotation model)
logisticCV <- cv.glmnet(FeatTrng, as.vector(OutTrng), lambda=costs,
                          family="binomial", alpha=0, nfolds=10)
if (verbose) {
  cat(' *** best lambda = ',logisticCV$lambda.min,' *** \n\n', sep='')
}


## Compute a P(FR | G) for all data
postprobTest <- predict(logisticCV, FeatTest, s="lambda.min", type="response")


## Train RIVER on training data
emModel <- integratedEM(FeatTrng, OutTrng, logisticCV$lambda.min,
          logisticCV$glmnet.fit, pseudoc,
          theta_init, costs, verbose)


## Compute P(FR | G, E)
dup.post <- testPosteriors(FeatTest, OutTest1, emModel)


## Check performance of models with N2 pairs
RIVER.roc <- roc(OutTest2, dup.post$posterior[,2]) # RIVER
GAM.roc <- roc(OutTest2, as.numeric(postprobTest)) # GAM

if (verbose) {
  cat('*** AUC (GAM - genomic annotation model): ',round(GAM.roc$auc,3),
    '\n    AUC (RIVER): ',round(RIVER.roc$auc,3),'\n     P-value: ',
      format.pval(roc.test(RIVER.roc, GAM.roc)$p.value,digits=2,eps=0.001),
      '***\n\n')
}

evaROC <-
  list(RIVER_sens=RIVER.roc$sensitivities,
         RIVER_spec=RIVER.roc$specificities,
         RIVER_auc=RIVER.roc$auc[1],
         GAM_sens=GAM.roc$sensitivities,
         GAM_spec=GAM.roc$specificities,
         GAM_auc=GAM.roc$auc[1],
         pvalue=roc.test(RIVER.roc, GAM.roc)$p.value)
class(evaROC) <- "eval"


#saveRDS(emModel, paste0(river_run_dir, stem, "_pvalue_thresh_", pvalue_threshold,".rds"))

#evaROC <- readRDS(paste0(river_run_dir, stem, "_pvalue_thresh_", pvalue_threshold,".rds"))


roc_figure_file_name <- paste0(river_run_dir, stem, "_pvalue_thresh_", pvalue_threshold, "_roc.pdf")
plot_roc(evaROC, pvalue_threshold,roc_figure_file_name)


beta_figure_file_name <- paste0(river_run_dir, stem, "_pvalue_thresh_", pvalue_threshold, "_beta_barplot.pdf")
plot_betas(emModel$beta, pvalue_threshold, beta_figure_file_name)

