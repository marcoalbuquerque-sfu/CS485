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

logit <- function(x) {
   return (1 / (1 + exp(-x)));
    }

update_weight <- function(data, weights) {

    X <- t(cbind(rep(1, nrow(data$data)), data$data));

    tmp <- weights %*% X;

    p <- logit(tmp);

    q <- 1 - p;

    W <- diag(as.numeric(p * q));

    y <- as.numeric(as.factor(data$class)) -1;

    new_weights <- weights + (solve(X %*% W %*% t(X)) %*% (X %*% t(y - p)))

    return(as.numeric(new_weights));

    }

calculate_class <- function(data, weights) {

    X <- t(cbind(rep(1, nrow(data$data)), data$data));
    tmp <- weights %*% X
    p <- logit(tmp);
    return (p > 0.5);

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

    train$class <- train$class[,1]

    e <- 0.0000000001

    weights_initial <- rep(1, ncol(train$data) + 1);
    weights_next <- rep(e, ncol(train$data) + 1);
    while (!all(weights_initial - weights_next < e)) {
        weights_initial <- weights_next;
        weights_next <- update_weight(train, weights_initial);
        }

    train$weights <- weights_next;

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

    a <- as.numeric(as.factor(train$class))-1;

    class <- data.frame(
        '1' = unique(train$class[!!a]),
        '0' = unique(train$class[!a])
        );

    colnames(class) <- c('1', '0');

    test$pred <- as.numeric(class[,as.character(as.numeric(calculate_class(test, train$weights)))]);

    if (!is.null(args$class)) {
        cat ( sum( test$class == test$pred ) / length(test$pred) );
        cat ("\n");
        }

    }