#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
})

read_input <- function() {
  input_text <- suppressWarnings(paste(readLines(con = file("stdin")), collapse = "\n"))
  if (nchar(input_text) == 0) {
    stop("No input JSON provided to run_rmvp_pseudo_qtn_selection.R")
  }
  fromJSON(input_text)
}

input <- read_input()

repo_root <- normalizePath(".")
rmvp_root <- normalizePath(file.path(repo_root, "..", "rMVP"))
source(file.path(rmvp_root, "R", "MVP.FarmCPU.r"))

# Provide fallbacks for bigmemory helpers when not available
if (!exists("is.big.matrix")) {
  is.big.matrix <- function(x) FALSE
}
if (!exists("deepcopy")) {
  deepcopy <- function(...) stop("deepcopy invoked without bigmemory support")
}

map_df <- data.frame(
  SNP = input$map$snp_id,
  CHROM = as.numeric(input$map$chrom),
  POS = as.numeric(input$map$pos),
  stringsAsFactors = FALSE
)
map_mat <- as.matrix(map_df)

pvalues <- as.numeric(input$pvalues)
if (length(pvalues) != nrow(map_mat)) {
  stop("Length of pvalues must match number of markers in map")
}

if (is.null(input$bin_sizes)) {
  bin_sizes <- c(5e5, 5e6, 5e7)
} else {
  bin_sizes <- as.numeric(input$bin_sizes)
}

iteration <- as.integer(input$iteration)
if (iteration == 2) {
  bin_size_use <- bin_sizes[length(bin_sizes)]
} else if (iteration == 3) {
  bin_size_use <- bin_sizes[length(bin_sizes) - 1]
} else {
  bin_size_use <- bin_sizes[1]
}

n_individuals <- as.integer(input$n_individuals)
if (is.na(n_individuals) || n_individuals <= 0) {
  stop("n_individuals must be provided and > 0")
}

if (is.null(input$bound)) {
  bound <- round(sqrt(n_individuals) / sqrt(log10(n_individuals)))
  if (bound < 1) bound <- 1
} else {
  bound <- as.integer(input$bound)
}

qtn_threshold <- as.numeric(input$qtn_threshold)
if (is.null(input$p_threshold)) {
  p_threshold <- NA_real_
} else {
  p_threshold <- as.numeric(input$p_threshold)
}

prev_qtns <- as.integer(input$prev_qtns)
if (length(prev_qtns) == 0) {
  prev_qtns <- NULL
} else {
  prev_qtns <- prev_qtns + 1
}

ld_threshold <- if (is.null(input$ld_threshold)) 0.7 else as.numeric(input$ld_threshold)

GP <- cbind(map_mat, P = pvalues, NA, NA, NA)
mySpecify <- FarmCPU.Specify(GI = map_mat, GP = GP, bin.size = bin_size_use, inclosure.size = bound)
seqQTN <- which(mySpecify$index == TRUE)

if (iteration == 2) {
  min_p <- suppressWarnings(min(pvalues, na.rm = TRUE))
  if (!is.na(p_threshold)) {
    if (min_p > p_threshold) {
      seqQTN <- NULL
    }
  } else {
    nm <- nrow(map_mat)
    if (min_p > 0.01 / nm) {
      seqQTN <- NULL
    }
  }
}

if (!is.null(prev_qtns)) {
  if (is.null(seqQTN)) {
    seqQTN <- prev_qtns
  } else {
    seqQTN <- sort(unique(c(seqQTN, prev_qtns)))
  }
}

if (!is.null(seqQTN)) {
  seqQTN.p <- pvalues[seqQTN]
  if (iteration == 2) {
    keep_idx <- which(seqQTN.p < qtn_threshold & !is.na(seqQTN.p))
    seqQTN <- seqQTN[keep_idx]
    seqQTN.p <- seqQTN.p[keep_idx]
  } else {
    keep_mask <- seqQTN.p < qtn_threshold
    if (!is.null(prev_qtns)) {
      keep_mask[seqQTN %in% prev_qtns] <- TRUE
    }
    keep_mask[is.na(seqQTN.p)] <- FALSE
    seqQTN <- seqQTN[keep_mask]
    seqQTN.p <- seqQTN.p[keep_mask]
  }
}

if (!is.null(seqQTN) && length(seqQTN) > 0) {
  order_idx <- order(seqQTN.p)
  seqQTN <- seqQTN[order_idx]
  seqQTN.p <- seqQTN.p[order_idx]
}

if (!is.null(seqQTN) && length(seqQTN) > 0) {
  geno <- matrix(as.numeric(unlist(input$genotype)), nrow = n_individuals, byrow = FALSE)
  if (ncol(geno) != nrow(map_mat)) {
    stop("Genotype matrix must have same number of columns as markers")
  }
  gdp_index <- seq_len(n_individuals)
  remove_res <- FarmCPU.Remove(GDP = geno, GDP_index = gdp_index, GM = map_mat, seqQTN = seqQTN, seqQTN.p = seqQTN.p, threshold = ld_threshold)
  seqQTN <- remove_res$seqQTN
} else {
  seqQTN <- numeric(0)
}

output <- list(seqQTN = as.integer(seqQTN - 1))
cat(toJSON(output, auto_unbox = TRUE))
