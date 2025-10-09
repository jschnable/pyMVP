library(rMVP)
library(readr)
library(data.table)
library(dplyr)
library(tidyverse)

# Set working directory to the analysis folder
setwd("/home/james/projects/AICodeProjects/HarshitaTroubleshooting/sorghum_gwas_analysis")

# First, extract the VCF file from the zip archive
system("unzip -o ../sap_mio_imputed_maf0.05_het0.1.renamed.zip")

# Data preparation using rMVP
MVP.Data(fileVCF="sap_mio_imputed_maf0.05_het0.1.renamed.vcf",
         filePhe="../sorghum_analysis_final_for_gwas_pc_scores_for_gwas.csv",
         out="sorghum_analysis",
         priority="speed",
         sep.phe=",",
         fileKin=TRUE,
         filePC=TRUE)

# Load the processed data
phenotype <- read.table("sorghum_analysis.phe", head=TRUE)
genotype <- attach.big.matrix("sorghum_analysis.geno.desc")
map <- read.table("sorghum_analysis.geno.map", head = TRUE)
Kinship <- attach.big.matrix("sorghum_analysis.kin.desc")
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix("sorghum_analysis.pc.desc"))

# Print data summary
cat("Phenotype data dimensions:", dim(phenotype), "\n")
cat("Number of markers:", nrow(map), "\n")
cat("Number of individuals with kinship data:", nrow(Kinship), "\n")
cat("Principal components dimensions:", dim(Covariates_PC), "\n")

# Resampling-based FarmCPU GWAS (100 iterations with 10% missing data each time)
for(x in 1:100){

  cat("Running bootstrap iteration:", x, "\n")

  phe1 = phenotype # make a copy of phenotype

  nline = nrow(phe1)

  # Randomly choose 10% phenotype to be NA
  phe1[sample(c(1:nline), as.integer(nline*0.1)), 2:ncol(phe1)] = NA

  # Rename the phenotype by attaching bootstrap number
  colnames(phe1) = paste0(colnames(phenotype), x)

  # Run FarmCPU for PC1 and PC2 only
  for(i in 2:3){

    trait_name <- colnames(phe1)[i]
    cat("  Processing trait:", trait_name, "\n")

    imMVP <- MVP(phe = phe1[,c(1,i)],
                 geno = genotype,
                 map = map,
                 K = Kinship,
                 CV.FarmCPU = Covariates_PC,
                 file.output = "pmap.signal",
                 nPC.FarmCPU = 3,
                 maxLoop = 10,
                 method = "FarmCPU",
                 threshold = 0.15,
                 p.threshold = 5e-07)
  }
}

# Function to summarize the occurrence of signals across bootstrap iterations
get.support = function(trait){

  files = list.files(pattern = paste0(trait,".*FarmCPU_signals.csv"))

  if (length(files) >= 1){

    signals <-
      files %>%
      map_df(~{
        df <- read.csv(., skip=1, header=F, stringsAsFactors = FALSE)
        # Select columns: SNP, CHROM, POS, A1, A2, MAF, Effect, SE, pvalue (9th column)
        df[,c(1:9)]
      })

    header <- c("SNP","CHROM","POS","A1","A2","MAF","Effect","SE","pvalue")
    colnames(signals) = header

    cat("Summary for trait", trait, ":\n")
    print(summary(signals))

    colnames(signals)[9] <- "pvalue"

    signals = signals %>%
      group_by(SNP, CHROM, POS) %>%
      summarise(P = mean(pvalue), support = n()/100, .groups = 'drop')

    write.table(signals, file=paste0("Z", trait, "signals.csv"),
                quote = F, row.names = F, sep=",")

    cat("Results written to:", paste0("Z", trait, "signals.csv"), "\n")

  } else {

    print(paste0("File not found for trait: ", trait))

  }
}

# Get trait names from the phenotype data (only PC1 and PC2)
trait_names <- c("PC1", "PC2")
cat("Available traits:", paste(trait_names, collapse=", "), "\n")

# Process support for each trait
for(trait in trait_names){
  cat("Processing support for trait:", trait, "\n")
  get.support(trait)
}

cat("FarmCPU resampling analysis completed!\n")
