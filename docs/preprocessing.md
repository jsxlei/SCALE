# Preprocessing

## Read alignment and peak calling  
We used [Kundaje pipeline](https://github.com/kundajelab/atac_dnase_pipelines) for aligning pair ended reads to hg19 and removing duplicates as [scABC](https://github.com/SUwonglab/scABC/). 

    bds_scr [SCR_NAME] atac.bds -align -species hg19 -species_file [SPECIES_FILE_PATH] -nth [NUM_THREADS] -fastq1_1 [READ_PAIR1] -fastq1_2 [READ_PAIR2]

The resulting bam files were merged using samtools. 

    samtools merge [AGGREGATE_BAM] *.trim.PE2SE.nodup.bam

The merged bam file was then used as input into MACS2 to call merged peaks for later analysis.

    bds_scr [SCR_NAME] atac.bds -species hg19 -species_file [SPECIES_FILE_PATH] -nth [NUM_THREADS] -se -filt_bam [AGGREGATE_BAM]


## Preprocessing
scATAC-seq data is required in peak count matrix as inputs.
We can import scATAC-seq preprocessing function in scale module to preprocessing as [scABC](https://github.com/SUwonglab/scABC/)

    from scale.utils import sample_filter, peak_filter, cell_filter
    
We filtered peaks that presented in >= 10 cells with >= 2 reads by peak_filter function 

    data = peak_filter(data)
    
We filtered cells that >= number of peaks / 50 by cell_filter function 

    data = cell_filter(data)
    
Or combine peak_filter and cell_filter 

    data = sample_filter(data)

