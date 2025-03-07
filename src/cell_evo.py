#%%
# The Bacteria Cell Class
""" Not Used anymore
class BacterialCell:
    ""
    Define the bacterial cell genome and antibiotics production
    - genes:
    - A: ABR
    - G: Growth
    - F: Fragile
    - Locus allele
        - Array with size = no. A whithin genome
        - the order on genome correspond to the order in the array
    ""
    genome = []
    antibiotics = []
    id = 0
    ancestors = []
    location = (0,0)


    def __init__(self,location,ancestors = [],gen_random = False, genome_size = 20, genome = None):
        self.genome = []
        self.antibiotics = []
        self.location = location
        if gen_random:
            self.genome, self.antibiotics = self.generate_random_genome(genome_size)
        else:
            self.genome = np.array(genome)
            for gene in self.genome:
                if gene == 'A':
                    self.antibiotics.append(AntibioticsGene(gen_random=True))

        self.id = np.random.randint(0, 0xFFFFFF)  # Generate random hex number up to 24 bits
        self.ancestors = ancestors.copy()
        self.ancestors.append(self.id) # add the ancestral history for further analysis

    def generate_random_genome(self, genome_size):
        self.genome = np.random.choice(['A', 'G', 'F'], genome_size, p=[0.5, 0.4, 0.1]) # ABR, Growth, Fragile, adjust individual generation frequencies here
        self.antibiotics = []
        for gene in self.genome:
            if gene == 'A':
                self.antibiotics.append(AntibioticsGene(gen_random=True))

        return self.genome, self.antibiotics
    
    def born_cell(self, location):
        cell = BacterialCell(location, self.ancestors, genome=self.genome)
        cell.genome = self.genome.copy()
        cell.antibiotics = self.antibiotics.copy()
        return cell
    
    def get_id(self):
        return self.id
    
    def get_location(self):
        return self.location

    def get_genome(self):
        return self.genome

    def get_genome_size(self):
        return len(self.genome)

    def get_antibiotics(self):
        # This method return the antibiotics object
        return self.antibiotics

    def get_ABR(self):
        # This method return the text representation of the antibiotics
        ABR = []
        for gene in self.antibiotics:
            ABR = np.append(ABR, gene.get_antibiotics())
        return ABR
    
    def get_growth(self):
        return self.genome.count('G')

    def get_ancestors(self):
        return self.ancestors
    
    def count_genes(self, gene):
        return np.count_nonzero(self.genome == gene)

    def get_gene_locus(self, gene):
        return np.where(self.genome == gene)[0]
"""

# This create a class EvoSpace that will hold the population of cells
class EvoSpace:
    genome_field = None
    antibiotics_field = None
    cell_list = None
    parameter_list = ["L", "tau_s", "eta", "alpha_g", "h_g", "beta_r", "alpha_a", "h_a", "r_a", "beta_g", "mu_d", "mu_a", "mu_f", "mu_n", "mu", "p_m"]

    def __init__(self, size):
        self.genome_field = np.empty(size, dtype=object)
        self.antibiotics_field = np.empty(size, dtype=object)
        self.cell_list = []
        self.parameters = {key: None for key in self.parameter_list}



    def set_parameters(self, parameters):
        missing_parameters = []
        for parameter in self.parameter_list:
            if parameter not in parameters:
                missing_parameters.append(parameter)
        if len(missing_parameters) > 0:
            print(f"Error: Missing parameters {missing_parameters}")
            return -1
        for i, (key, value) in enumerate(parameters.items()):
            if key in self.parameter_list:
                self.parameters[key] = value
            else:
                print(f"Error: Invalid parameter {key}")
                return -1
        self.radius_vectors = self.calculate_radius_vectors(self.parameters['r_a'])
        return 0

    def add_cell(self, cell):
        if self.get_cell(cell.get_location()) is not None:
            return -1
        self.cell_list.append(cell)
        self.genome_field[cell.get_location()] = cell.get_genome()
        # self.antibiotics_field[cell.get_location()] = cell.get_ABR()

    def generate_random_cells(self, n_cells, genome_size):
        for i in range(n_cells):
            location = (np.random.randint(0, self.genome_field.shape[0]), np.random.randint(0, self.genome_field.shape[1]))
            while self.get_cell(location) is not None:
                location = (np.random.randint(0, self.genome_field.shape[0]), np.random.randint(0, self.genome_field.shape[1]))
            cell = BacterialCell(location, gen_random=True, genome_size=genome_size, antibiotics_length = parameters['mu'], field = self)
            self.add_cell(cell)

    def remove_cell(self, cell, clear_antibiotics = False):
        self.cell_list.remove(cell)
        self.genome_field[cell.get_location()] = None
        if clear_antibiotics:
            self.antibiotics_field[cell.get_location()] = None

    def add_antibiotics(self, location, antibiotics:list):
        if self.antibiotics_field[location] is None:
            self.antibiotics_field[location] = np.array(antibiotics)
        else:

            self.antibiotics_field[location] = np.append(self.antibiotics_field[location], antibiotics)
            # self.antibiotics_field[location] = np.unique(self.antibiotics_field[location])
        return self.antibiotics_field[location]


    def get_cell(self, location):
        return self.genome_field[location[0], location[1]]

    def get_cell_list(self):
        return self.cell_list

    def get_genome_field(self):
        return self.genome_field

    def get_antibiotics_field(self):
        return self.antibiotics_field

    def get_cell_count(self):
        return len(self.cell_list)

    def plot_genome_field(self, ax):
        figure = np.zeros(self.genome_field.shape)
        for cells in self.cell_list:
            figure[cells.get_location()] = cells.get_ansestors()[-1]
        # for i in range(figure.shape[0]):
        #     for j in range(figure.shape[1]):
        #         try:
        #             figure[i,j] = self.genome_field[i,j].get_ancestors()[-1]
        #         except:
        #             figure[i,j] = 0
        ax.imshow(figure, cmap='gist_ncar')
        ax.set_title('Genome field')

    def plot_antibiotics_field(self, ax):
        figure = np.zeros(self.genome_field.shape)
        # for cells in self.cell_list:
        #     if self.antibiotics_field[cells.get_location()] is not None:
        #         abs = np.array(self.antibiotics_field[cells.get_location()])
        #         print(abs)
        #         try:
        #             figure[cells.get_location()] = len(set(abs))
        #         except:
        #             figure[cells.get_location()] = 1
                
        #     else:
        #         figure[cells.get_location()] = 0
            # figure[cells.get_location()] = len(set(self.antibiotics_field[cells.get_location()]))
        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                try:
                    figure[i,j] = len(set(self.antibiotics_field[i,j]))
                except:
                    figure[i,j] = 0

        im = ax.imshow(figure, cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title('Antibiotics field')
        ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    def plot_ancestral_field(self,ax):
        figure = np.zeros(self.genome_field.shape)
        for cells in self.cell_list:
            figure[cells.get_location()] = cells.get_ancestors()[0]

        # for i in range(figure.shape[0]):
        #     for j in range(figure.shape[1]):
        #         # try:
        #         #     figure[i,j] = self.genome_field[i,j].get_ancestors()[0]
        #         #     print(figure[i,j])
        #         # except:
        #         #     figure[i,j] = 0
        #         cell = self.get_cell((i,j))
        #         if cell is not None:
        #             figure[i,j] = cell.get_ancestors()[0]
        ax.imshow(figure, cmap='gist_ncar')
        ax.set_title('Ancestor field')
        ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)



    
    def show_population(self):
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        print(f"Total NO. of Cells in System is {self.get_cell_count()}")
        self.plot_ancestral_field(axs[0])
        self.plot_antibiotics_field(axs[1])
        fig.show()
        return fig

    def calculate_radius_vectors(self,radius):
        # Returns a list of all  position vectors within a given radius
        # of the origin. The radius is the maximum distance from the origin
        radius_vectors = []
        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                if i**2+j**2 <= radius**2:
                    radius_vectors.append((i,j))
        return radius_vectors


class AntibioticsGene:
    """
    The antibiotics type is a 16 bit number
    - 0x0000: No antibiotics
    """

    
    def __init__(self, antibiotics_type = 0x0000, gen_random = False, antibiotics_length = 16):
        if gen_random:
            self.antibiotics_length = antibiotics_length
            self.generate_random_antibiotics(antibiotics_length)
        else:
            self.antibiotics_type = antibiotics_type
    
    def generate_random_antibiotics(self, antibiotics_length = 16):
        self.antibiotics_type = np.random.randint(0, 2**antibiotics_length)

    def get_antibiotics(self):
        return self.antibiotics_type
    
    def set_antibiotics(self, antibiotics_type):
        self.antibiotics_type = antibiotics_type
    
    def uniform_mutate(self, antibiotics_length = 16):
        """
        Uniform mutation
        Each bit has a 50% chance of being flipped by the random mutation
        """
        mutation = np.random.randint(0, 2**antibiotics_length)
        self.antibiotics_type = self.antibiotics_type ^ mutation
    
    def flip_bit(num: int, bit_position: int) -> int:
        return num ^ (1 << bit_position)

    def normal_mutate(self, antibiotics_length = 16):
        """
        Normal mutation
        No of bit flipped is a random number from a normal distribution
        """
        # per bit mutation rate

        # different masks
        # select random number of masks and apply to flip bits
    
        n_mutation = np.random.normal(antibiotics_length/2, antibiotics_length/4)
        n_mutation = int(n_mutation)
        if n_mutation < 0:
            n_mutation = 0
        if n_mutation > antibiotics_length:
            n_mutation = antibiotics_length
        for i in range(n_mutation):
            bit_position = np.random.randint(0, antibiotics_length)
            self.antibiotics_type = AntibioticsGene.flip_bit(self.antibiotics_type, bit_position)
        
        return n_mutation

class BacterialCell:
    """
    Properties:
    - genome: A list of genes
    - antibiotics: A list of antibiotics genes
    - id: A unique identifier
    - ancestors: A list of the id of the ancestors
    - location: A tuple of the location of the cell in the field

    Methods:
    - __init__: Initialize the cell
        - gen_random: Generate a random genome
        - genome_size: The size of the genome
        - genome: The defined genome of the cell

    - set_parameters: Set the parameters of the cell
    - get_parameters: Get the parameters of the cell

    - born_cell: Create a new cell with the same genome and antibiotics
    - generate_random_genome: Generate a random genome

    - cell_copy: Copy the cell to a new location
    - cell_replication: Replicate the cell to its neighborhood
    - cell_death: Kill the cell
    - cell_antibiotics_production: Produce antibiotics
    - cell_movement: Move the cell to a new location

    - gene_duplications: Duplicate a gene in the genome
    - gene_deletions: Delete a gene in the genome
    - fragile_deletion: delete genes to the right of the fragile locus
    - fragile_generation: generate fragile genes
    - antibiotics_mutation: Mutate the antibiotics
    - genome_mutation: Mutate the genome, using the methods above


    - get_id: Get the id of the cell
    - get_location: Get the location of the cell
    - get_genome: Get the genome of the cell
    - get_genome_size: Get the size of the genome
    - get_antibiotics: Get the antibiotics of the cell in objects
    - get_ABR: Get the antibiotics in text form
    - get_ancestors: Get the ancestors of the cell
    - count_genes: Count the number of genes in the genome
    - get_gene_locus: Get the locus of a gene in the genome



    - calc_antibiotics_resistance: Calculate the antibiotics resistance on the block
    - calc_rep_prob: calculate the reproduction probability of the cell
    - calc_death_prob: calculate the death probability of the cell
    - calc_ab_production_prob: calculate the antibiotics production probability
    - hamming_distance: calculate the hamming distance between two bit strings
    """

    

    parameter_list = ["alpha_g", "h_g", "beta_r", "alpha_a", "h_a", "r_a", "beta_g", "mu_d", "mu_a", "mu_f", "mu_n", "mu", "p_m"]

    def __init__(self,location : tuple,field,ancestors = [],gen_random = False, genome_size = 20, genome = [], antibiotics_length = 16, ):
        """
        Initialize a new cell instance.

        Parameters:
        location (tuple): The location of the cell.
        ancestors (list, optional): List of ancestor cells. Defaults to an empty list.
        gen_random (bool, optional): Flag to generate a random genome. Defaults to False.
        genome_size (int, optional): Size of the genome to generate if gen_random is True. Defaults to 20.
        genome (list, optional): Predefined genome sequence. Defaults to None.

        Attributes:
        genome (list): The genome sequence of the cell.
        
        location (tuple): The location of the cell.
        id (int): Unique identifier for the cell.
        """
        size = max(len(genome), genome_size)
        self.genome = np.empty((size,2), dtype=object)
        
        self.location = location
        if gen_random:
            raw_genome = self.generate_random_genome(genome_size) # generate random genome if gen_random is True
        else:
            raw_genome = np.array(genome)
            
        for i, gene in enumerate(raw_genome):
                self.genome[i,:] = [gene, None]
                if gene == 'A':
                    self.genome[i] = ('A',AntibioticsGene(gen_random=True, antibiotics_length=antibiotics_length))            

        self.id = np.random.randint(0, 0xFFFFFF) # Generate random hex number up to 24 bits
        self.ancestors = ancestors.copy()
        self.ancestors.append(self.id) # add the ancestral history for further analysis

        self.parameters = {key: None for key in self.parameter_list}
        self.field = field

        self.set_parameters(self.field.parameters)

    def born_cell(self, location):
        """
        Create a new BacterialCell instance at the given location.

        Args:
            location (tuple): The coordinates where the new cell will be born.

        Returns:
            BacterialCell: A new instance of BacterialCell with the same genome and antibiotics as the parent.
        """
        new_cell = BacterialCell(location, ancestors = self.ancestors, genome=self.get_genome(), antibiotics_length = self.parameters['mu'], field = self.field)
        new_cell.genome = np.array([[gene, ab] for gene, ab in self.genome if gene is not None])
        return new_cell

    def generate_random_genome(self, genome_size : int, gene_freq = [0.5, 0.4, 0.1]):
        """
        Generates a random genome and associated antibiotics genes.

        This method generates a random genome of specified size using the nucleotides 'A', 'G', and 'F' with respective probabilities of 0.5, 0.4, and 0.1. It also creates a list of antibiotics genes for each 'A' nucleotide in the genome.

        Args:
            genome_size (int): The size of the genome to generate.

        Returns:
            A list consisting raw_genome with no type attached
        """
        genome = np.random.choice(['A', 'G', 'F'], genome_size, p=gene_freq) # ABR, Growth, Fragile, adjust individual generation frequencies here

        return genome
    def check_parameters(self, parameter, range):
        """
        Check if the parameter value is within the specified range.

        Args:
            parameter (str): The parameter to check.
            range (tuple): The valid range of values for the parameter.

        Returns:
            bool: True if the parameter value is within the specified range, False otherwise.
        """
        if parameter in self.parameters:
            if self.parameters[parameter] < range[0] or self.parameters[parameter] > range[1]:
                return False
        return True

    def set_parameters(self, parameters):
        missing_parameters = []
        for parameter in self.parameter_list:
            if parameter not in parameters:
                missing_parameters.append(parameter)
        if len(missing_parameters) > 0:
            print(f"Error: Missing parameters {missing_parameters}")
            return -1
        for i, (key, value) in enumerate(parameters.items()):
            if key in self.parameter_list:
                if value < 0:
                    print(f"Error: Invalid parameter value {key}")
                    return -1
                self.parameters[key] = value
            # else:
            #     print(f"Error: Invalid parameter {key}")
            #     return -1
        return 0

    def cell_copy(self, evo_field, cell, new_location:tuple, mutation = False):
        """
        Creates a copy of a cell and places it in a new location within the evolutionary field.

        Args:
            evo_field: The evolutionary field where the new cell will be placed.
            cell: The cell to be copied.
            new_location (tuple): The coordinates where the new cell will be placed.
            mutation (bool, optional): If True, the new cell will undergo a genome mutation. Defaults to False.

        Returns:
            The newly created cell.
        """
        new_cell = self.born_cell(new_location)
        if mutation:
            new_cell.genome_mutation()
        evo_field.add_cell(new_cell)

        return new_cell

    def cell_replication(self, evo_field, mutation: bool = False):
        """
        Replicates a cell in the evolutionary field.
        This method attempts to replicate a given cell in one of its neighboring locations
        within the evolutionary field. The neighboring locations are shuffled to ensure
        random placement. If a valid empty location is found, a new cell is created as a 
        copy of the original cell, with an optional mutation.
        Parameters:
        evo_field (EvolutionField): The evolutionary field containing the cells.
        cell (Cell): The cell to be replicated.
        mutation (bool, optional): If True, the new cell will have a mutation. Defaults to False.
        Returns:
        Cell: The newly created cell if replication is successful, otherwise None.
        """
        neighbourhood_vectors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        np.random.shuffle(neighbourhood_vectors)
        field_size = evo_field.genome_field.shape
        for vector in neighbourhood_vectors:
            new_location = (self.location[0] + vector[0], self.location[1] + vector[1])
            if new_location[0] >= 0 and new_location[0] < field_size[0] and new_location[1] >= 0 and new_location[1] < field_size[1]:
                if evo_field.get_cell(new_location) is None:
                    # Create a new cell at the empty location
                    new_cell = self.cell_copy(evo_field, self, new_location, mutation)
                    break
                else:
                    new_cell = None
        
        return new_cell

    def cell_death(self, evo_field):
        """
        Removes a cell from the evolutionary field.
        Parameters:
        evo_field (EvolutionField): The evolutionary field from which the cell will be removed.
        cell (Cell): The cell to be removed from the evolutionary field.
        Returns:
        None
        """

        evo_field.remove_cell(self)
        return

    def antibiotics_production(self, evo_field, radius_vectors):
        """
        Simulates the production of antibiotics by a cell.
        Parameters:
        evo_field (EvoField): The evolutionary field where the cell resides.
        cell (Cell): The cell that produces the antibiotics.
        radius_vectors (list of tuple): List of vectors representing the radius around the cell where antibiotics are produced.
        Returns:
        None
        """
        try:
            ab_gene = np.random.choice(self.get_ABR())
            
            for vector in radius_vectors:
                location = (self.location[0] + vector[0], self.location[1] + vector[1])
                if location[0] >= 0 and location[0] < evo_field.genome_field.shape[0] and location[1] >= 0 and location[1] < evo_field.genome_field.shape[1]:
                    evo_field.add_antibiotics(location, ab_gene)
        except:
            pass
            # print("Error: No antibiotics genes found")

        return

    def cell_movement(self, evo_field):
        """
        Moves a cell to a new location within the evolutionary field.
        Parameters:
        evo_field (EvolutionField): The evolutionary field containing the cells.
        cell (Cell): The cell to be moved.
        Returns:
        Cell: The cell after movement.
        """
        neighbourhood_vectors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        np.random.shuffle(neighbourhood_vectors)
        field_size = evo_field.genome_field.shape
        for vector in neighbourhood_vectors:
            new_location = (self.location[0] + vector[0], self.location[1] + vector[1])
            if new_location[0] >= 0 and new_location[0] < field_size[0] and new_location[1] >= 0 and new_location[1] < field_size[1]:
                if evo_field.get_cell(new_location) is None:
                    evo_field.remove_cell(self)
                    self.location = new_location
                    evo_field.add_cell(self)
                    break
        
        return cell

    def gene_duplications(self, locus):
        """
        Duplicates a gene in the genome at the specified locus.
        Parameters:
        locus (int): The index of the gene to be duplicated.
        Returns:
        self.genome (list): The updated genome after gene duplication.
        """
        insertion_site = np.random.randint(0, len(self.genome))
        self.genome = np.insert(self.genome, insertion_site, self.genome[locus])
        return self.genome

    def gene_deletions(self, locus):
        """
        Deletes a gene in the genome at the specified locus.
        Parameters:
        locus (int): The index of the gene to be deleted.
        Returns:
        self.genome (list): The updated genome after gene deletion.
        """
        self.genome = np.delete(self.genome, locus)
        return self.genome

    def fragile_deletion(self, fragile_locus):
        """
        Deletes genes to the right of the fragile locus.
        Parameters:
        fragile_locus (int): The index of the fragile gene.
        Returns:
        self.genome (list): The updated genome after fragile deletion.
        """
        if self.genome[fragile_locus][0] == 'F':
            self.genome = self.genome[:fragile_locus]
        else:
            pass
            # print("Error: Invalid fragile locus")
        
        return self.genome

    def fragile_generation(self):
        """
        Generates fragile genes to the right of the fragile locus.
        Parameters:
        fragile_locus (int): The index of the fragile gene.
        Returns:
        self.genome (list): The updated genome after fragile generation.
        """
        try:
            locus = np.random.choice(self.get_gene_locus('F'))
        except:
            return self.genome
        insertion_site = np.random.randint(locus, len(self.genome))
        self.genome = np.insert(self.genome, insertion_site, ('F', None))
        return self.genome

    def antibiotics_mutation(self, locus):
        """
        Mutates the antibiotics genes in the genome.
        Parameters:
        None
        Returns:
        self.genome (list): The updated genome after antibiotics mutation.
        """
        if self.genome[locus][0] == 'A':
            self.genome[locus][1].uniform_mutate()
    
        return self.genome
    
    def genome_mutation(self, mu_a = 0.005, mu_d = 0.001, mu_f = 0.01, mu_n = 0.01):
        """
        Mutates the genome of the cell.
        Parameters:
        None
        Returns:
        self.genome (list): The updated genome after mutation.
        """
        n_A = self.count_genes('A')
        n_G = self.count_genes('G')
        n_F = self.count_genes('F')
        n_genes = self.get_genome_size()
        genome = self.get_genome()
        i = 0
        while i < len(genome) - 1:
            # every gene have a chance equal to the mutation rate to mutate
            gene = genome[i]
            i += 1
            if gene == 'A' and np.random.rand() < mu_a:
                self.antibiotics_mutation(i)
            elif np.random.rand() < mu_d:
                self.gene_deletions(i)
            elif np.random.rand() < mu_d:
                self.gene_duplications(i)
            elif np.random.rand() < mu_f:
                self.fragile_deletion(i)
            genome = self.get_genome()

            

        if np.random.rand() < mu_n: # generate new fragile gene only once
            self.fragile_generation()

        return self.genome

    def get_id(self):
        """
        Get the id of the cell.
        Parameters:
        None
        Returns:
        id (int): The unique identifier of the cell.
        """
        return self.id

    def get_location(self):
        """
        Get the location of the cell.
        Parameters:
        None
        Returns:
        location (tuple): The coordinates of the cell.
        """
        return self.location

    def get_genome(self):
        """
        Get the genome of the cell.
        Parameters:
        None
        Returns:
        genome (list): The genome sequence of the cell.
        """
        try:
            genes = self.genome[:,0]
        except:
            return []
        return genes

    def get_genome_size(self):
        """
        Get the size of the genome.
        Parameters:
        None
        Returns:
        size (int): The number of genes in the genome.
        """
        return len(self.genome)
    
    def get_antibiotics(self):
        """
        Get the antibiotics genes of the cell.
        Parameters:
        None
        Returns:
        antibiotics (list): The antibiotics genes of the cell.
        """
        ab_locus = self.get_gene_locus('A')
        if len(ab_locus) == 0:
            return []
        antibiotics = self.genome[ab_locus][:,1]
        return antibiotics

    def get_ABR(self):
        """
        Get the antibiotics resistance of the cell.
        Parameters:
        None
        Returns:
        ABR (list): The antibiotics resistance of the cell.
        """
        ABR = []
        for gene in self.get_antibiotics():
            ABR = np.append(ABR, gene.get_antibiotics())
        return ABR

    def get_ancestors(self):
        """
        Get the ancestors of the cell.
        Parameters:
        None
        Returns:
        ancestors (list): The ancestors of the cell.
        """
        return self.ancestors

    def count_genes(self, gene):
        """
        Count the number of genes in the genome.
        Parameters:
        gene (str): The gene to count in the genome.
        Returns:
        count (int): The number of occurrences of the gene in the genome.
        """
        genome = self.get_genome()
        if len(genome) == 0:
            return 0
        count = len(np.where(genome == gene)[0])
        return count

    def get_gene_locus(self, gene):
        """
        Get the locus of a gene in the genome.
        Parameters:
        gene (str): The gene to search for in the genome.
        Returns:
        locus (list): The indices of the gene in the genome.
        """
        try:
            locus = np.where(np.array([gene[0] for gene in self.genome]) == gene)
        except:
            locus = []
        return locus


    def calc_antibiotics_resistance(self, evo_field, beta_r):
        """
        Calculate the antibiotics resistance on the block.
        Parameters:
        evo_field (EvoField): The evolutionary field containing the cells.
        radius (int): The radius of the block to consider.
        Returns:
        resistance (float): The resistance of the block.
        """
        antibiotics = self.get_ABR()
        s = 0
        block_antibiotics = evo_field.get_antibiotics_field()[self.location[0], self.location[1]]

        for ab in antibiotics:
            min_d = 16
            if block_antibiotics is not None:
                try:
                    for block_ab in block_antibiotics:
                        d = self.hamming_distance(ab, block_ab)
                        if d < min_d:
                            min_d = d
                except:
                    min_d = self.hamming_distance(ab, block_antibiotics)
            else:
                min_d = 0
            s += min_d

        R = np.exp(-beta_r * s * s)
        return R

    def calc_rep_prob(self, evo_field, alpha_g, h_g, beta_r):
        """
        Calculate the reproduction probability of the cell.
        Parameters:
        evo_field (EvoField): The evolutionary field containing the cells.
        alpha_g (float): The growth rate.
        h_g (float): The half growth number.
        beta_r (float): The antibiotics resistance factor.
        Returns:
        rep_prob (float): The reproduction probability of the cell.
        """

        n_G = self.count_genes('G')
        R = self.calc_antibiotics_resistance(evo_field, beta_r)
        rep_prob = alpha_g * n_G * R / ( h_g + n_G )

        return rep_prob

    def calc_death_prob(self, evo_field, beta_r):
        """
        Calculate the death probability of the cell.
        Parameters:
        evo_field (EvoField): The evolutionary field containing the cells.
        beta_r (float): The antibiotics resistance factor.
        Returns:
        death_prob (float): The death probability of the cell.
        """
        R = self.calc_antibiotics_resistance(evo_field, beta_r)
        death_prob = (1 - R) *0.1

        return death_prob

    def calc_ab_production_prob(self, evo_field, alpha_a, h_a, beta_g):
        """
        Calculate the antibiotics production probability.
        Parameters:
        evo_field (EvoField): The evolutionary field containing the cells.
        beta_a (float): The antibiotics production factor.
        h_a (float): The half antibiotics number.
        beta_g (float): The growth factor.
        Returns:
        ab_prod_prob (float): The antibiotics production probability.
        """
        n_A = self.count_genes('A')
        n_G = self.count_genes('G')
        ab_prod_prob = alpha_a * n_A / ( h_a + n_A ) * np.exp(-beta_g * n_G)

        return ab_prod_prob

    def hamming_distance(self,a,b):
        # Calculate the hamming distance between two bit strings

        return bin(int(a)^int(b)).count('1')

    def cell_tick(self):
        """
        Run the cell for one time step.
        Parameters:
        None
        Returns:
        None
        """
        evo_field = self.field
        pars = self.parameters
        abr = self.calc_antibiotics_resistance(evo_field, pars['beta_r'])
        rep_prob = self.calc_rep_prob(evo_field, pars['alpha_g'], pars['h_g'], pars['beta_r'])
        death_prob = self.calc_death_prob(evo_field, pars['beta_r'])
        ab_prod_prob = self.calc_ab_production_prob(evo_field, pars['alpha_a'], pars['h_a'], pars['beta_g'])
        if np.random.rand() < rep_prob:
            self.cell_replication(evo_field, mutation = True)
        if np.random.rand() < death_prob:
            pass
            # self.cell_death(evo_field)
        if np.random.rand() < ab_prod_prob:
            self.antibiotics_production(evo_field, evo_field.radius_vectors)
        return None

        
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('src')
parameters = {
    "L": (10,10),
    "tau_s": 1000,
    "eta": 0.001,
    "alpha_g": 0.5,
    "h_g": 5,
    "beta_r": 0.2,
    "alpha_a": 2,
    "h_a": 2,
    "r_a": 5,
    "beta_g": 0.5,
    "mu_d": 0.01,
    "mu_a": 0.05,
    "mu_f": 0.05,
    "mu_n": 0.01,
    "mu": 16,
    "p_m": 0.1
}

space = EvoSpace((200,200))
space.set_parameters(parameters)
space.generate_random_cells(10, int(0.01 * 200 * 200))
# space.add_cell(BacterialCell((0,0),gen_random=True, genome_size=5))
fig = space.show_population()
fig.savefig('/home/roger/MCB/strepto_evolution/output/population_plot.png')


cell1 = space.cell_list[0]
cell1.cell_replication(space, mutation = True)
print(cell1.count_genes('A'))
print(cell1.count_genes('G'))
print(cell1.count_genes('F'))
print(cell1.get_genome())
import timeit
def run(n):
    for i in range(n):
        try:
            cell = np.random.choice(space.cell_list)
        except:
            break
        rep = cell.calc_rep_prob(space, space.parameters['alpha_g'], space.parameters['h_g'], space.parameters['beta_r'])
        death = cell.calc_death_prob(space, space.parameters['beta_r'])
        ab_prod = cell.calc_ab_production_prob(space, space.parameters['alpha_a'], space.parameters['h_a'], space.parameters['beta_g'])
        # print(rep, death, ab_prod)
        if np.random.rand() < rep:
            cell.cell_replication(space, mutation = True)
        if np.random.rand() < death:
            cell.cell_death(space)
        if np.random.rand() < ab_prod:
            cell.antibiotics_production(space, space.radius_vectors)
        # cell.cell_replication(space)
        # cell.antibiotics_production(space, space.radius_vectors)
run(10000)

# for cell in space.cell_list:
#     for i in range(10):
#         cell.antibiotics_production(space, space.radius_vectors)

# for cell in space.cell_list:
#     abr = cell.calc_antibiotics_resistance(space, space.parameters['beta_r'])
#     rep = cell.calc_rep_prob(space, space.parameters['alpha_g'], space.parameters['h_g'], space.parameters['beta_r'])
#     death = cell.calc_death_prob(space, space.parameters['beta_r'])
#     abpro = cell.calc_ab_production_prob(space, space.parameters['alpha_a'], space.parameters['h_a'], space.parameters['beta_g'])
#     print(abr, rep, death, abpro)
# cell1.cell_death(space)
fig = space.show_population()
fig.savefig('/home/roger/MCB/strepto_evolution/output/population_plot2.png')

# %%
# cell1 = BacterialCell((0,0),gen_random=True, genome_size=5)
# print(cell1.get_ABR())
# cell1.set_parameters(parameters)
# print(cell1.get_genome())
# for i in range(10):
#     cell1 = cell1.born_cell((0,0))
# print(cell1.get_genome())
# cell1.cell_tick(parameters)

# %%