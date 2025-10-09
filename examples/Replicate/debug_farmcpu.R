library(rMVP)
setwd("examples/Replicate")
phenotype <- read.table("py_check.phe", head=TRUE)
genotype <- attach.big.matrix("py_check.geno.desc")
map <- read.table("py_check.geno.map", head=TRUE)
Kinship <- attach.big.matrix("py_check.kin.desc")
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix("py_check.pc.desc"))
res <- MVP(phe = phenotype[,c("Taxa","PC1")],
           geno = genotype,
           map = map,
           K = Kinship,
           CV.FarmCPU = Covariates_PC,
           file.output = FALSE,
           nPC.FarmCPU = 3,
           maxLoop = 10,
           method = "FarmCPU",
           threshold = 0.15,
           p.threshold = 5e-07,
           verbose = FALSE)
print(str(res$farmcpu.results))
