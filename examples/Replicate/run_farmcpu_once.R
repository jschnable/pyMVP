library(rMVP)
setwd("examples/Replicate")

phenotype <- read.table("py_check.phe", header=TRUE)
genotype <- attach.big.matrix("py_check.geno.desc")
map <- read.table("py_check.geno.map", header=TRUE)
Kinship <- attach.big.matrix("py_check.kin.desc")
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix("py_check.pc.desc"))

MVP(phe = phenotype[,c("Taxa","PC1")],
    geno = genotype,
    map = map,
    CV.FarmCPU = Covariates_PC,
    K = Kinship,
    file.output = TRUE,
    out = "py_check_run",
    nPC.FarmCPU = 3,
    maxLoop = 10,
    method = "FarmCPU",
    p.threshold = 5e-07,
    threshold = 0.15,
    verbose = FALSE)

# Read the result file
sig_file <- "py_check_run.FarmCPU.csv"
res <- read.csv(sig_file, header=TRUE)
idx <- which(res$SNP == "Chr01-24886605")
if(length(idx) > 0) {
  cat("Pvalue Chr01-24886605:", res$P.value[idx], "\n")
} else {
  cat("SNP Chr01-24886605 not found\n")
}
cat("Top 5 entries:\n")
print(head(res[order(res$P.value), c("SNP","P.value")], 5))
