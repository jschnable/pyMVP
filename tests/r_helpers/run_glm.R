#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
})

read_input <- function() {
  input_text <- suppressWarnings(paste(readLines(con = file("stdin")), collapse = "\n"))
  if (nchar(input_text) == 0) {
    stop("No input JSON provided to run_glm.R")
  }
  fromJSON(input_text)
}

silent_source <- function(path) {
  tryCatch({
    suppressWarnings(suppressMessages(source(path)))
    TRUE
  }, error = function(e) {
    message(sprintf("Warning: failed to source %s: %s", path, e$message))
    FALSE
  })
}

silent_require <- function(pkg) {
  suppressWarnings(suppressMessages(
    tryCatch({
      library(pkg, character.only = TRUE)
      TRUE
    }, error = function(...) FALSE)
  ))
}

input <- read_input()
repo_root <- normalizePath("..")

gapit_loaded <- silent_source(file.path(repo_root, "gapit_functions.txt"))
farmcpu_loaded <- silent_source(file.path(repo_root, "FarmCPU_functions.txt"))

if (!gapit_loaded || !farmcpu_loaded) {
  pkg_candidates <- c("GAPIT", "GAPIT3")
  pkg_loaded <- FALSE
  for (pkg in pkg_candidates) {
    if (silent_require(pkg)) {
      pkg_loaded <- TRUE
      break
    }
  }
  if (!pkg_loaded) {
    stop("Failed to load GAPIT helper functions: source files missing and GAPIT/GAPIT3 package unavailable")
  }
  if (!exists("FarmCPU.LM")) {
    for (pkg in pkg_candidates) {
      if (requireNamespace(pkg, quietly = TRUE)) {
        fn <- get0("FarmCPU.LM", envir = asNamespace(pkg), inherits = FALSE)
        if (!is.null(fn)) {
          FarmCPU.LM <- fn
          break
        }
      }
    }
  }
  if (!exists("FarmCPU.Prior")) {
    for (pkg in pkg_candidates) {
      if (requireNamespace(pkg, quietly = TRUE)) {
        fn_prior <- get0("FarmCPU.Prior", envir = asNamespace(pkg), inherits = FALSE)
        if (!is.null(fn_prior)) {
          FarmCPU.Prior <- fn_prior
          break
        }
      }
    }
  }
}

if (!exists("FarmCPU.LM")) {
  stop("FarmCPU.LM function unavailable after loading helpers")
}

if (exists("FarmCPU.Prior") && !"kinship.algorithm" %in% names(formals(FarmCPU.Prior))) {
  FarmCPU.Prior_original <- FarmCPU.Prior
  FarmCPU.Prior <- function(GM, P = NULL, Prior = NULL, kinship.algorithm = "FARM-CPU") {
    FarmCPU.Prior_original(GM = GM, P = P, Prior = Prior)
  }
}

phenotype_obj <- input$phenotype
if (is.null(phenotype_obj)) {
  stop("Phenotype input missing")
}

extract_phenotype_matrix <- function(obj) {
  if (is.matrix(obj)) {
    return(obj)
  }
  if (is.data.frame(obj)) {
    return(as.matrix(obj))
  }
  if (is.list(obj)) {
    mat <- do.call(rbind, lapply(obj, unlist))
    return(mat)
  }
  stop("Unsupported phenotype structure: ", paste(class(obj), collapse = ";"))
}

Y_mat <- extract_phenotype_matrix(phenotype_obj)
if (is.null(dim(Y_mat)) || ncol(Y_mat) < 2) {
  stop("Phenotype input malformed: expected >=2 columns, got ", ncol(Y_mat))
}
Y_df <- data.frame(
  ID = as.character(Y_mat[, 1]),
  Trait = as.numeric(Y_mat[, 2])
)

n_ind <- nrow(Y_df)
GD_mat <- matrix(as.numeric(unlist(input$genotype)), nrow = n_ind, byrow = TRUE)

if (!is.null(input$covariates)) {
  cov_obj <- input$covariates
  if (is.list(cov_obj) && !is.data.frame(cov_obj)) {
    CV_mat <- as.matrix(do.call(rbind, cov_obj))
  } else {
    CV_mat <- as.matrix(cov_obj)
  }
  mode(CV_mat) <- "numeric"
} else {
  CV_mat <- NULL
}

if (!is.null(input$impute_value)) {
  fill_val <- as.numeric(input$impute_value)
  GD_mat[GD_mat == -9] <- fill_val
}

res <- FarmCPU.LM(
  y = Y_df$Trait,
  GDP = GD_mat,
  w = CV_mat,
  orientation = "col",
  model = "A",
  ncpus = 1
)

pvals <- as.numeric(res$P[, ncol(res$P)])
cat(toJSON(list(pvalues = pvals), auto_unbox = TRUE))
