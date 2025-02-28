# Strepto Antibiotics Evolution

https://doi.org/10.15252/msb.202211353

## Description of the system

*Streptomyces coelicolor*

-  Generate somatic cells through mutations rather than gene regulation
-  mutation induce division of labor
-  multicellular bacteria 
-  grow from haploid spores
-  Life cycle
   -  mycelium -> spores
-  when differentiating to produce spores, colonies produce antibiotics
-  antibiotics regulate strain competition
-  10^5 antibiotics are synthesized, each species carry 30 biosynthetic gene clusters
-  gene clusters encode antibiotics or resistance
-  different cell types produce spores and ABR
-  The antibiotic‐hyperproducing subset of cells in *S. coelicolor* colonies arises due to massive and irreversible deletions at the left and right arms of the *Streptomyces* linear chromosome 
-  larger deletion -> more antibiotics, less spores
-  ABR cells generated independently 

## Hypothesis

Here, we hypothesize that chromosomal gene order has evolved such that  some functional groups of genes have localized at the telomeric ends of  the chromosome, making them more susceptible to deletion due to genome  instability. By this argument, genome instability becomes adaptive  within the context of this genome organization, because it facilitates  the generation of sterile antibiotic‐producing mutants from replicating  cells. We show that a genome architecture capable of generating a  mutation‐driven division of labor evolves in a computational model of  antibiotic‐producing bacterial colonies.

## Description of the model

-  We focus on the vegetative growth stage, especially during the developmental transition to sporulation 
-  1st phase
   -  colonies grow, interact & compete
-  2nd phase
   -  secondary metabolism
   -  antibiotics diffuse around colonies
   -  colony claim new space
-  3rd phase
   -  sporulation when resource limits

### World model

-  2d surface
   -  antibiotics diffuse
   -  cells grow
-  Bacteria replicate locally into empty lattices site in direct neighborhood
-  Evolution through multiple rounds of 3 phase growth 
-  colonies develop for tau_s
   -  each step lattices updated in random order
-  Sporulation happen after tau_s (assume nutrients depleted) (why not cumulative cell number?)
   -  randomly select a small fraction 𝜉 of the bacteria
   -  spores seed next growth cycle
-  end the growth cycle by killing at random a fraction 1−𝜉 of all the cells, leaving the remaining cells to initiate the next growth cycle.

### Evolution Model

-  
-  

### Genome

-  We model the *Streptomyces* linear chromosome with a beads‐on‐a‐string model
-  Genes
   -  Growth
   -  ABP
   -  Fragile (Mutation hotspots)
-  A genome can contain a variable number of genes and fragile sites.
-  a bacterium with no growth‐promoting genes remains alive, but it is assigned a growth rate equal to zero.
-  Growth/ ABP purely depend on genotype
-  deletion of part of the genome leads both to reduced growth and to increased antibiotic production
-  Growth‐promoting genes inhibit the expression of antibiotic genes (i.e., regulation is fixed), and this inhibition is lifted when growth genes  are in low numbers. 
-  $k_{repl} = \alpha_g R_g / (g+h_g)$
   -  $\alpha_g$ max replication probability per unit time = 0.1
   -  $R_g$ specify antibiotics resistance
   -  $g$ number of growth genes
   -  $h_g$ no. growth gene producing half growth rate 
   -  Michaelis menten kinetics
-  $k_{ab} = \alpha_aA(a)exp(-\beta_gg)$
   -  $\alpha_a$ max antibiotics production rate
   -  $A(a)$ Function mapping no. antibiotics gene to rate
   -  $\beta_g$ scaling factor for inhibition
   -  $g$ no. of growth genes
-  type of antibiotics determined by bit string
   -  2^16 types 
   -  16 bit long 
-  Antibiotics secreted to neighbor
   -  radius $r_a$ = 10 lattice
-  multiple antibiotics can be present at the same lattice
-  bacteria are resistant to ABR if they encode same 16 bit string
-  Resistance decrease when difference increases
-  Cells die and remove from lattice if they have no resistance
-  Mutations (happens once a cell divide)
   -  duplication & deletion event
      -  $\mu_d$ 
      -  duplication & deletion of a locus (where will be the offspring gene be?)
   -  fragile event
      -  $\mu_f$
      -  cause deletion of genome to its right
   -  new fragile
      -  $\mu_n$ 
      -  produce new fragile site per genome
   -  antibiotics mutation
      -  $\mu_a$
      -  change antibiotics type
   -   

![image-20250227143459123](./Strepto Antibiotics Evolution.assets/image-20250227143459123.png)



### Program design

-  generate a genome field
-  generate an antibiotics field