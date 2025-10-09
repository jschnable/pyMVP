library(rMVP)
setwd("examples/Replicate")
phenotype <- read.table("py_check.phe", head=TRUE)
genotype <- attach.big.matrix("py_check.geno.desc")
map <- read.table("py_check.geno.map", head = TRUE)
Kinship <- attach.big.matrix("py_check.kin.desc")
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix("py_check.pc.desc"))

runs <- 10
threshold <- 5e-07
snp_target <- "Chr09-57005333"
idx <- which(map$SNP == snp_target)
if(length(idx)==0){stop("SNP not found")}

pvals <- numeric(runs)
set.seed(123)
for (x in 1:runs) {
  phe1 <- phenotype
  nline <- nrow(phe1)
  mask_idx <- sample(1:nline, as.integer(nline*0.1))
  phe1[mask_idx, 2:ncol(phe1)] <- NA
  colnames(phe1) <- colnames(phenotype)
  res <- MVP(phe = phe1[,c("Taxa","PC1")],
             geno = genotype,
             map = map,
             K = Kinship,
             CV.FarmCPU = Covariates_PC,
             file.output = FALSE,
             nPC.FarmCPU = 3,
             maxLoop = 10,
             method = "FarmCPU",
             threshold = 0.15,
             p.threshold = threshold,
             verbose = FALSE)
  farmcpu <- res$farmcpu.results
  if (is.null(farmcpu)) {
    pvals[x] <- NA
  } else {
    pvals[x] <- farmcpu[idx,3]
  }
  cat("Run", x, "pvalue", pvals[x], "\n")
}
cat("Support count:", sum(pvals <= threshold, na.rm=TRUE), "\n")
cat("Mean p (hits):", mean(pvals[pvals <= threshold], na.rm=TRUE), "\n")
write.table(pvals, file="py_check_r_pvals.txt", row.names=FALSE, col.names=FALSE)
