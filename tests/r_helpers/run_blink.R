#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
})

read_input <- function() {
  input_text <- suppressWarnings(paste(readLines(con = file("stdin")), collapse = "\n"))
  if (nchar(input_text) == 0) {
    stop("No input JSON provided to run_blink.R")
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
repo_root <- normalizePath(".")
project_root <- normalizePath(file.path(repo_root, ".."))

gapit_loaded <- silent_source(file.path(project_root, "gapit_functions.txt"))
farmcpu_loaded <- silent_source(file.path(project_root, "FarmCPU_functions.txt"))

if (!silent_require("BLINK")) {
  stop("The BLINK R package is not installed or loadable")
}

# Ensure FarmCPU helpers accept the kinship.algorithm argument used by Blink.R
if (!"kinship.algorithm" %in% names(formals(FarmCPU.Prior))) {
  FarmCPU.Prior_original <- FarmCPU.Prior
  FarmCPU.Prior <- function(GM, P = NULL, Prior = NULL, kinship.algorithm = "FARM-CPU") {
    FarmCPU.Prior_original(GM = GM, P = P, Prior = Prior)
  }
}

Y_df <- as.data.frame(input$phenotype)
colnames(Y_df) <- c("ID", "Trait")
Y_df$ID <- as.character(Y_df$ID)
Y_df$Trait <- as.numeric(Y_df$Trait)

n_ind <- nrow(Y_df)
GD_mat <- matrix(as.numeric(unlist(input$genotype)), nrow = n_ind, byrow = FALSE)

map_df <- data.frame(
  SNP = as.character(input$map$snp_id),
  CHROM = as.numeric(input$map$chrom),
  POS = as.numeric(input$map$pos),
  stringsAsFactors = FALSE
)

if (!is.null(input$covariates)) {
  CV_mat <- matrix(as.numeric(unlist(input$covariates)), nrow = n_ind, byrow = FALSE)
} else {
  CV_mat <- NULL
}

if (!is.null(input$prior)) {
  prior_df <- data.frame(
    SNP = as.character(input$prior$snp),
    Chr = as.numeric(input$prior$chrom),
    Pos = as.numeric(input$prior$pos),
    Weight = as.numeric(input$prior$weight),
    stringsAsFactors = FALSE
  )
} else {
  prior_df <- NULL
}

blink_args <- list(
  Y = Y_df,
  GD = GD_mat,
  GM = map_df,
  CV = CV_mat,
  Prior = prior_df,
  maxLoop = ifelse(is.null(input$maxLoop), 10, as.integer(input$maxLoop)),
  converge = ifelse(is.null(input$converge), 1, as.numeric(input$converge)),
  LD = ifelse(is.null(input$ld_threshold), 0.7, as.numeric(input$ld_threshold)),
  maf.threshold = ifelse(is.null(input$maf_threshold), 0, as.numeric(input$maf_threshold)),
  method.sub = ifelse(is.null(input$method_sub), "reward", input$method_sub),
  p.threshold = ifelse(is.null(input$p_threshold), NA_real_, as.numeric(input$p_threshold)),
  cutOff = ifelse(is.null(input$qtn_threshold), 0.01, as.numeric(input$qtn_threshold)),
  iteration.output = FALSE,
  file.output = FALSE,
  time.cal = FALSE,
  stepwise = FALSE,
  GP = NULL
)

captured_output <- capture.output({
  blink_result <- suppressMessages(suppressWarnings(do.call(Blink, blink_args)))
})

parse_iteration_log <- function(lines) {
  log_entries <- list()
  current_iter <- NA_integer_
  expect_seq <- FALSE
  for (line in lines) {
    clean_line <- trimws(sub("^\\[[0-9]+\\]\\s*", "", line))
    if (grepl("Iteration:", clean_line, fixed = TRUE)) {
      match_iter <- sub('.*Iteration: *([0-9]+).*', '\\1', clean_line)
      if (grepl('^[0-9]+$', match_iter)) {
        current_iter <- as.integer(match_iter)
      }
      expect_seq <- FALSE
    } else if (grepl("seqQTN", clean_line, fixed = TRUE)) {
      expect_seq <- TRUE
    } else if (expect_seq) {
      expect_seq <- FALSE
      if (grepl("NULL", clean_line, ignore.case = TRUE)) {
        indices <- integer(0)
      } else {
        indices <- suppressWarnings(scan(text = clean_line, what = integer(), quiet = TRUE))
        indices <- indices[!is.na(indices)]
      }
      log_entries[[length(log_entries) + 1]] <- list(
        iteration = ifelse(is.na(current_iter), NA_integer_, current_iter),
        selected_qtns = indices
      )
    }
  }
  log_entries
}

iteration_log <- parse_iteration_log(captured_output)

pvalues <- as.numeric(blink_result$GWAS[, "P.value"])
output <- list(
  pvalues = pvalues,
  iteration_qtns = iteration_log
)
cat(toJSON(output, auto_unbox = TRUE))
