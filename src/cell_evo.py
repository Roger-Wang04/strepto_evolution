import numpy as np



# Initialization related functions

def calculate_radius_vectors(radius):
    # Returns a list of all  position vectors within a given radius
    # of the origin. The radius is the maximum distance from the origin
    radius_vectors = []
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if i**2+j**2 <= radius**2:
                radius_vectors.append((i,j))
    return radius_vectors

# The antibiotics Class
class AntibioticsGene:
    """
    The antibiotics type is a 16 bit number
    - 0x0000: No antibiotics
    """

    
    def __init__(self, antibiotics_type = 0x0000, gen_random = False):
        if gen_random:
            self.generate_random_antibiotics()
        else:
            self.antibiotics_type = antibiotics_type
    
    def generate_random_antibiotics(self):
        self.antibiotics_type = np.random.randint(0, 2**16)

    def get_antibiotics(self):
        return self.antibiotics_type
    
    def set_antibiotics(self, antibiotics_type):
        self.antibiotics_type = antibiotics_type
    
    def uniform_mutate(self):
        """
        Uniform mutation
        Each bit has a 50% chance of being flipped by the random mutation
        """
        mutation = np.random.randint(0, 2**16)
        self.antibiotics_type = self.antibiotics_type ^ mutation
    
    def flip_bit(num: int, bit_position: int) -> int:
        return num ^ (1 << bit_position)

    def normal_mutate(self):
        """
        Normal mutation
        No of bit flipped is a random number from a normal distribution
        """
        # per bit mutation rate

        # different masks
        # select random number of masks and apply to flip bits
    
        n_mutation = np.random.normal(8, 4)
        n_mutation = int(n_mutation)
        if n_mutation < 0:
            n_mutation = 0
        if n_mutation > 16:
            n_mutation = 16
        for i in range(n_mutation):
            bit_position = np.random.randint(0, 16)
            self.antibiotics_type = AntibioticsGene.flip_bit(self.antibiotics_type, bit_position)
        
        return n_mutation


# The Bacteria Cell Class
class BacteriaCell:
    """
    Define the bacterial cell genome and antibiotics production
    - genes:
    - A: ABR
    - G: Growth
    - F: Fragile
    - Locus allele
        - Array with size = no. A whithin genome
        - the order on genome correspond to the order in the array
    """
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
        cell = BacteriaCell(location, self.ancestors, genome=self.genome)
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

# This create a class EvoSpace that will hold the population of cells
class EvoSpace:
    genome_field = None
    antibiotics_field = None
    cell_list = None

    def __init__(self, size):
        self.genome_field = np.empty(size, dtype=object)
        self.antibiotics_field = np.empty(size, dtype=object)
        self.cell_list = []

    def add_cell(self, cell):
        if get_cell(cell.get_location()) is not None:
            return -1
        self.cell_list.append(cell)
        self.genome_field[cell.get_location()] = cell.get_genome()
        self.antibiotics_field[cell.get_location()] = cell.get_ABR()

    def generate_random_cells(self, n_cells, genome_size):
        for i in range(n_cells):
            while self.get_cell(location) is not None:
                location = (np.random.randint(0, self.genome_field.shape[0]), np.random.randint(0, self.genome_field.shape[1]))
            cell = BacteriaCell(location, gen_random=True, genome_size=genome_size)
            self.add_cell(cell)

    def remove_cell(self, cell, clear_antibiotics = False):
        self.cell_list.remove(cell)
        self.genome_field[cell.get_location()] = None
        if clear_antibiotics:
            self.antibiotics_field[cell.get_location()] = None

    
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
        figure = np.zeros(field.shape)
        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                try:
                    figure[i,j] = self.genome_field[i,j].get_ancestors()[-1]
                except:
                    figure[i,j] = 0

    def plot_antibiotics_field(self, ax):
        figure = np.zeros(field.shape)
        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                try:
                    figure[i,j] = len(set(self.antibiotics_field[i,j]))
                except:
                    figure[i,j] = 0

    def plot_ancestral_field(self,ax):
        figure = np.zeros(field.shape)
        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                try:
                    figure[i,j] = self.genome_field[i,j].get_ancestors()[0]
                except:
                    figure[i,j] = 0
    
    def show_population(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        print(f"Total NO. of Cells in System is {self.get_cell_count()}")
        self.plot_ancestral_field(genome_field, axs[0])
        self.plot_antibiotics_field(antibiotics_field, axs[1])
        fig.show()

# Cell Behaviours

def cell_copy(evo_field, cell, new_location):
    new_cell = cell.born_cell(new_location)
    new_cell = 
    evo_field.add_cell(new_cell)
