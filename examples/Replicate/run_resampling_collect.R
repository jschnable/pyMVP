library(rMVP)
setwd("examples/Replicate")
phenotype <- read.table("py_check.phe", head=TRUE)
genotype <- attach.big.matrix("py_check.geno.desc")
map <- read.table("py_check.geno.map", head = TRUE)
Kinship <- attach.big.matrix("py_check.kin.desc")
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix("py_check.pc.desc"))

runs <- 10
threshold <- 5e-07
support <- rep(0, nrow(map))
p_sums <- rep(0, nrow(map))

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
  if (is.null(farmcpu)) next
  pvals <- farmcpu[,3]
  hits <- which(pvals <= threshold)
  support[hits] <- support[hits] + 1
  p_sums[hits] <- p_sums[hits] + pvals[hits]
}

support_freq <- support / runs
mean_p <- ifelse(support > 0, p_sums / support, NA)
output <- data.frame(SNP = map$SNP, CHROM = map$CHROM, POS = map$POS,
                     support = support_freq, mean_p = mean_p)
write.csv(output, file="py_check_resampling_summary.csv", row.names=FALSE)
