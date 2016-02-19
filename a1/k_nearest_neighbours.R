suppressMessages(library(argparse));
suppressMessages(library(flexclust));
parser <- ArgumentParser(description="Train and Test using K-Nearest Neighbours Algorithm");

parser$add_argument(
    "--mode", "-m",
    choices=c('train', 'test'),
    required=TRUE,
    help="Train or Test a K-Nearest Neighbours Model"
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

parser$add_argument(
	"-k",
	type="integer",
	help="Number of Neighbours to visit to determine class assignment"
	);

args <- parser$parse_args();

findksmallestindex <- function(vector, k) {
    n <- length(vector);

    output <- rep(NA, k);
    current_index <- 1;
    na_remain <- k;
    while ( any(is.na(output)) ) {
        smallest <- sort(vector, partial=current_index)[current_index];
        truth <- smallest == vector;
        if ( na_remain >= sum(truth) ) {
            output[current_index:(current_index + sum(truth) - 1)] <- which(truth);
            current_index <- current_index + sum(truth);
            na_remain <- na_remain - sum(truth);
            } 
        else if ( na_remain < sum(truth) ) {
            output[current_index:(current_index + na_remain - 1 )] <- sample(which(truth), na_remain, replace=F);
            current_index <- length(output);
            na_remain <- 0;
            }
        }
    return(output);

    }

assignclass <- function(indexes, classes) {
    data <- table(classes[indexes]);
    max <- max(data);
    index <- which(max == data);
    if (length(index) > 1) {
        return(names(data)[sample(index, 1)])
        }
    else {
        return(names(data)[index])
        }
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

    save(train, file = paste0(args$output_prefix, ".RData") );

    }

if (args$mode == "test") {
    
    if ( is.null(args$output_prefix) ) {
        stop("You must specify --output_prefix to store class assignment and or testing summary");
        }

    if ( is.null(args$k) ) {
        stop("You must specify -k to determine number of neighbours to visit");
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
        }
    else {
        test$class <- read.table(       
            args$class,
            sep=",",
            header=FALSE
            )
        }

    distance <- dist2(train$data, test$data, method="euclidean");
    kclosest <- apply(distance, 2, findksmallestindex, args$k);
    class <- vector();
    
    if ( args$k == 1) {
        class <- sapply(kclosest, assignclass, train$class);
        }
    else {
        class <- apply(kclosest, 2, assignclass, train$class);
        }

    write.table(
      	x = class,
        file = paste0(args$output_prefix, ".test_class.csv"),
        quote=F,
        row.names = FALSE,
        col.names = FALSE
        )

    if (!is.null(args$class)) {
        cat ( sum( test$class == class ) / length(class));
        cat("\n");
        }
    }