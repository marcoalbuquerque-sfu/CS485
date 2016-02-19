suppressMessages(library(argparse));
library(MASS);
parser <- ArgumentParser(description="Train and Test using Linear Regression with Regularization");

parser$add_argument(
    "--mode", "-m",
    choices=c('train', 'test'),
    required=TRUE,
    help="Train or Test a Linear Regression Model"
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
	"--lambda", "-l",
	type="double",
	help="Lambda variable for Regularization"
	);

args <- parser$parse_args();

calculateA <- function(x) {
    a <- c(1, as.numeric(x));
    return( t(t(a)) %*% t(a) );
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

    if ( is.null(args$lambda) ) {
        stop("You must specify --lambda for regularization of linear regression");
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
        )[,1];

    train$lambda <- args$lambda;
    train$A <- matrix( apply( apply( train$data, 1, calculateA ), 1, sum), nrow=( length(train$data[1,]) + 1), byrow=T );
    train$lambdaI <- diag(length(train$data[1,]) + 1) * args$lambda;
    train$A_plus_lambdaI <- train$A + train$lambdaI;
    train$inverse_A_plus_lambdaI <- solve(train$A_plus_lambdaI);
    train$b <- apply( cbind( rep(1, nrow(train$data) ), train$data ) * train$class, 2, sum); 
    train$w <- train$inverse_A_plus_lambdaI %*% train$b;

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
        }
    else {
        test$class <- read.table(       
            args$class,
            sep=",",
            header=FALSE
            )
        }

    test$class <- test$class[,1];

    class <- rep(0, length(test$class));

    for (i in 1:nrow(test$data)) {
        class[i] <- sum( train$w[2:length(train$w)] * test$data[i,] ) + train$w[1];
        #data_i <- test$data[i,];
        #A <- calculateA(data_i);
        #lambdaI <- diag(length(test$data[1,]) + 1) * train$lambda;
        #tmp <- A + lambdaI;
        #b <- tmp %*% train$w;
        #class[i] <- ginv(c(1,as.numeric(test$data[i,]))) %*% b;}
        }

    ### RETURN RESULTS AND ACCURACY

    SS_tot <- var(test$class) * length(test$class);

    SS_res <- 0

    for (i in 1:length(test$class)) {
        SS_res <- SS_res + ( (test$class[i] - class[i]) ^ 2 );
        }

    cat(1 - (SS_res / SS_tot));
    cat("\n");

    }