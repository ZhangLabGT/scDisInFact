# FROM ubuntu:latest
# RUN apt-get update && \
#       apt-get -y install sudo

# RUN sudo apt install -y dirmngr gnupg apt-transport-https ca-certificates software-properties-common\
# && sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9\
# && sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'\
# && sudo apt -y install r-base libcurl4-openssl-dev libssl-dev libxml2-dev\
# && R --version

# RUN Rscript -e "update.packages(ask = FALSE)"\
# && Rscript -e "install.packages('BiocManager')"\
# && Rscript -e "BiocManager::install('IRanges')"\
# && Rscript -e "BiocManager::install('Biobase')"\
# && Rscript -e "BiocManager::install('GenomicRanges')"\
# # && Rscript -e "BiocManager::install('GenomeInfoDb')"\
# && Rscript -e "BiocManager::install('SummarizedExperiment')"\
# && Rscript -e "install.packages('devtools')"\
# && Rscript -e "devtools::install_github('YosefLab/SymSim', upgrade_dependencies = TRUE, dependencies = TRUE)"

# save file https://docs.docker.com/storage/
# build -t name directory, -f allows outside build context
# sudo docker build -f Dockerfile -t symsim .
# sudo docker run -v xxxxx:xxxxx [-it] [--rm] symsim

FROM ziqipeter/symsim:latest
# run script
RUN mkdir /symsim2/
COPY simulation_condition_impute.R /symsim_test/simulation_condition_impute.R
CMD Rscript /symsim_test/simulation_condition_impute.R
# sudo docker build -f Dockerfile -t symsim_test .
# sudo docker run -v /localscratch/ziqi/scDisInFact/data/simulated/imputation_10000_500_0.3_20_1:/symsim_test/dataset --rm symsim_test2 &


