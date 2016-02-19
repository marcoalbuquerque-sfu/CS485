suppressMessages(library(argparse));
suppressMessages(library(flexclust));
parser <- ArgumentParser(description="Train and Test using K-Nearest Neighbours Algorithm");

parser$add_argument(
    "--mode", "-m",
    choices=c('train', 'test'),
    required=TRUE,
    help="Train or Test a Mixture of Gaussians model"
	);

parser$add_argument(
    "--rdata", "-r",
    help="Model RData generated from train mode, requierd for testing"
	);

parser$add_argument(
    "--output_prefix", "-op",
    help="Output prefix"
	);

parser$add_argument(
    "--input", "-i",
    help="Sample by Feature Matrix"
	);

parser$add_argument(
    "--class", "-c",
    help="Sample by Class Matrix, All Samples in input must be in class matrix"
	);

args <- parser$parse_args();



calculate_class_means <- function(data) {
    
    classes <- unique(data$class);

    class_means <- matrix(
        rep(0, length(classes) * ncol(data$data)),
        ncol = ncol(data$data)
        );

    rownames(class_means) <- classes;

    for ( class in classes ) {

        indexes <- which(data$class == class);
        class_means[as.character(class),] <- apply(data$data[indexes,], 2, mean);

        }

    return(class_means);

    }

calculate_covariance_matrix <- function(data) {

    classes <- unique(data$class);

    sigma <- matrix(
        rep(0, ncol(data$data) * ncol(data$data)),
        ncol = ncol(data$data)
        );

    class_means <- calculate_class_means(data);

    for ( class in classes ) {

        indexes <- which(data$class == class);

        a <- apply(
            t(t(data$data[indexes,]) - class_means[as.character(class),]), 
            1, 
            function(x) {as.numeric(x %*% t(x))}
            );

        b <- apply(a, 1, sum);

        c <- matrix(
            b,
            ncol = ncol(data$data),
            byrow=T
            );
        

        d <- ( 1 / nrow(data$data) ) * c;

        sigma <- sigma + d;

        }

    return(sigma);   

    }

calculate_probabilities <- function(train, test) {

    sigma_inv <- solve(train$covariance);
    classes <- as.character(unique(train$class));
    priors <- table(train$class) / length(train$class);
    probs <- apply(
        test$data, 
        1,
        function(x) {
            probs <- rep(0, length(classes));
            names(probs) <- classes;
            for (class in classes) {
                mu <- as.numeric(x - train$means[class,]);
                probs[class] <- exp(-.5 * (t(mu) %*% sigma_inv %*% mu)) * priors[class];
                }
            return (probs);
            }
        );

    sums <- apply(probs, 2, sum);

    return(t(probs) / sums);

    }

if ( args$mode == "train" ) {

    if ( is.null(args$output_prefix) ) {
        stop("You must specify --output_prefix to store model data");
        }

    if ( is.null(args$input) ) {
        stop("You must specify --input to load feature data");
        }

    if ( is.null(args$class) ) {
        stop("You must specify --class to train model");
        }

    train <- list();

    train$data <- read.table(
        args$input,
        sep=",",
        header=FALSE
    	);

    train$class <- read.table(
        args$class,
        sep=",",
        header=FALSE
    	);

    train$class <- train$class[,1];

    train$means <- calculate_class_means(train);

    train$covariance <- calculate_covariance_matrix(train);

    save(train, file = paste0(args$output_prefix, ".RData") );

    }

if (args$mode == "test") {
    
    if ( is.null(args$output_prefix) ) {
        stop("You must specify --output_prefix to store class assignment and or testing summary");
        }

    if ( is.null(args$rdata) ) {
        stop("You must specify --rdata for testing mode");
        }

    if ( is.null(args$input) ) {
        stop("You must specify --input to load feature data");
        }

    if ( is.null(args$class) ) {
        stop("You must specify --class to test data");
        }

    load(args$rdata);

    test <- list();

    test$data <- read.table(
        args$input,
        sep=",",
        header=FALSE
    	);

    if ( is.null(args$class) ) {
        test$class <- NA
    } else {
        test$class <- read.table(       
            args$class,
            sep=",",
            header=FALSE
            )
        }

    test$probs <- calculate_probabilities(train, test);

    test$pred <- colnames(test$probs)[apply(test$probs, 1, order, decreasing=T)[1,]];

    if (!is.null(args$class)) {
        cat ( sum( test$class == test$pred ) / length(test$pred) );
        cat ("\n");
        }
    }