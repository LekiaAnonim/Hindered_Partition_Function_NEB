import math
import numpy as np

def get_moment_of_inertia_about_binding_atom(ads, binding_atom_idx=None):
    """
    Calculate moment of inertia for rotation around z-axis through binding atom.
    
    Args:
        ads: ASE Atoms object
        binding_atom_idx: Index of binding atom (if None, uses lowest atom)
    
    Returns:
        Moment of inertia in amu*Angstrom^2
    """
    # Get binding atom position
    if binding_atom_idx is None:
        heights = ads.positions[:, 2]
        binding_atom_idx = np.argmin(heights)
    
    binding_pos = ads.positions[binding_atom_idx]
    center_xy = binding_pos[:2]  # xy position of rotation axis
    
    # Method 1: Direct calculation about the binding atom
    # This is more accurate for rotation about z-axis through binding atom
    masses = ads.get_masses()
    I_zz = 0.0
    
    for i, atom in enumerate(ads):
        m = masses[i]
        # Distance from atom to rotation axis (z-axis through binding atom)
        dx = atom.position[0] - center_xy[0]
        dy = atom.position[1] - center_xy[1]
        r_perp_squared = dx**2 + dy**2
        
        I_zz += m * r_perp_squared
    
    return I_zz

class HinderedTranslationPartitionFunction:
    """
    Calculate partition functions for 2D hindered translation of adsorbates on surfaces.
    Based on Hill's treatment of hindered translation.
    """
    
    # Physical constants
    H = 6.62607015e-34  # J*s (Planck's constant)
    K = 1.380649e-23  # J/K (Boltzmann constant)
    EV = 1.60218e-19  # J (electron volt conversion)
    AMU_TO_KG = 1.66053906660e-27  # kg
    ANGSTROM_TO_METER = 1e-10  # m
    
    def __init__(self, m, W_x, W_y, b, M, T=300):
        """
        Initialize hindered translation partition function calculator.
        
        Args:
            m: Mass of adsorbed species (in amu)
            W_x: Translational energy barrier height in x-direction (in eV)
            W_y: Translational energy barrier height in y-direction (in eV)
            b: Nearest neighbor distance between surface atoms (in Angstrom)
            M: Number of surface sites
            T: Temperature (in K), default 300 K
        """
        self.m = m * self.AMU_TO_KG  # Convert to kg
        self.W_x = W_x * self.EV  # Convert to J
        self.W_y = W_y * self.EV  # Convert to J
        self.b = b * self.ANGSTROM_TO_METER
        self.M = M
        self.T = T
        
        # Cached properties
        self._v_x = None
        self._r_x = None
        self._T_x = None
    
    def potential_energy(self, x, y, V_o=0):
        """
        Calculate potential energy for hindered xy-motion.
        
        Args:
            x: x-coordinate (in meters)
            y: y-coordinate (in meters)
            V_o: Potential energy at minima (in J)
        
        Returns:
            Potential energy V_xy (in J)
        """
        V_xy = V_o + (self.W_x/2) * (1 - math.cos(2*math.pi*x/self.b)) + \
               (self.W_y/2) * (1 - math.cos(2*math.pi*y/self.b))
        return V_xy
    
    @property
    def vibrational_frequency(self):
        """
        Calculate vibrational frequency at low temperatures.
        At low temperatures, the adsorbate vibrates about minima with frequency v_x = v_y.
        
        Returns:
            Vibrational frequency (in Hz)
        """
        if self._v_x is None:
            self._v_x = math.sqrt(self.W_x / (2 * self.m * (math.sqrt(3)/2) * self.b**2))
        return self._v_x
    
    @property
    def r_x(self):
        """
        Ratio of energy barrier height to vibrational frequency times Planck's constant.
        
        Returns:
            Dimensionless ratio r_x
        """
        if self._r_x is None:
            self._r_x = self.W_x / (self.H * self.vibrational_frequency)
        return self._r_x
    
    @property
    def T_x(self):
        """
        Dimensionless temperature T_x.
        
        Returns:
            Dimensionless temperature
        """
        if self._T_x is None:
            self._T_x = (self.K * self.T) / (self.H * self.vibrational_frequency)
        return self._T_x
    
    @staticmethod
    def I_0(x):
        """
        Zeroth-order modified Bessel function of the first kind, I_0(x).
        Uses series expansion approximation.
        
        Args:
            x: Argument
        
        Returns:
            I_0(x)
        """
        return 1 + (x**2)/4 + (x**4)/64 + (x**6)/2304 + (x**8)/147456 + (x**10)/14745600
    
    def q_classical(self):
        """
        Classical partition function.
        
        Returns:
            Classical partition function
        """
        return self.M * (math.pi * self.r_x * self.T_x) * \
               math.exp(-self.r_x / self.T_x) * self.I_0(self.r_x / (2 * self.T_x))
    
    def q_HO(self):
        """
        Quantum partition function for a single harmonic oscillator 
        independently in two identical directions.
        
        Returns:
            Harmonic oscillator partition function
        """
        return ((math.exp(-1/(2 * self.T_x))) / (1 - math.exp(-1/self.T_x)))**2
    
    def q_HO_classical(self):
        """
        Classical limit of harmonic oscillator partition function.
        
        Returns:
            Classical HO partition function
        """
        return self.T_x**2
    
    def q_xy(self):
        """
        Combined partition function.
        This gives the same result as Hill. Accurate at higher temperatures but 
        gives incorrect zero-point energy contribution at lower temperatures.
        
        Returns:
            Combined partition function
        """
        return (self.q_classical() * self.q_HO()) / self.q_HO_classical()
    
    def zero_point_energy_correction(self):
        """
        Calculate zero-point energy correction.
        
        Returns:
            Zero-point energy correction (in J)
        """
        return self.H * self.vibrational_frequency * (1 / (2 + 16 * self.r_x))
    
    def q_trans(self):
        """
        Total partition function for 2D translator with zero-point energy correction.
        The factor of 2 accounts for two independent directions (x and y) with 
        identical potential energy versus distance.
        
        Returns:
            Total translational partition function
        """
        return self.q_xy() * math.exp(2 * self.zero_point_energy_correction() / (self.K * self.T))
    
    def f_trans(self):
        """
        Normalized partition function per site.
        q_trans is divided by M since q_trans refers to the whole surface with M sites,
        whereas q_HO is the partition function for a single harmonic oscillator site.
        
        Returns:
            Normalized partition function
        """
        return (self.q_trans() / self.M) / self.q_HO()
    
    def P_trans(self):
        """
        Probability factor for translation.
        
        Returns:
            Translation probability factor
        """
        return self.q_xy() / (self.M * self.q_HO())


class HinderedRotorPartitionFunction:
    """
    Calculate partition functions for hindered rotation of adsorbates.
    Based on McClurg et al. treatment.
    """
    
    # Physical constants
    H = 6.62607015e-34  # J*s (Planck's constant)
    K = 1.380649e-23  # J/K (Boltzmann constant)
    EV = 1.60218e-19  # J (electron volt conversion)
    AMU_TO_KG = 1.66053906660e-27  # kg
    ANGSTROM_TO_METER = 1e-10  # m
    
    def __init__(self, W_r, n, I, T=300, symmetric_number=1, rotor_asymmetric=True):
        """
        Initialize hindered rotor partition function calculator.
        
        Args:
            W_r: Rotational energy barrier height (in eV)
            n: Number of equivalent minima in one full rotation
               (n=1 for heteronuclear, n=2 for homonuclear, n=3 for NH3, etc.)
            I: Reduced moment of inertia (in amu*Angstrom^2)
            T: Temperature (in K), default 300 K
            symmetric_number: Symmetry number (2 for homonuclear diatomic, 1 for heteronuclear)
            rotor_asymmetric: True for asymmetric rotor, False for symmetric rotor
        """
        self.W_r = W_r * self.EV  # Convert to J
        self.n = n
        self.I = I * self.AMU_TO_KG * (self.ANGSTROM_TO_METER**2)  # Convert to kg*m^2
        self.T = T
        self.symmetric_number = symmetric_number
        self.rotor_asymmetric = rotor_asymmetric
        
        # Cached properties
        self._v_r = None
        self._r_r = None
        self._T_r = None
    
    def potential_energy(self, phi):
        """
        Calculate potential energy for hindered rotation.
        
        Args:
            phi: Torsional angle (in radians)
        
        Returns:
            Potential energy (in J)
        """
        return (self.W_r / 2) * (1 - math.cos(self.n * phi))
    
    @property
    def vibrational_frequency(self):
        """
        Calculate vibrational frequency at low temperatures.
        At low temperatures, the adsorbate vibrates about minima with frequency v_r.
        
        Returns:
            Vibrational frequency (in Hz)
        """
        if self._v_r is None:
            self._v_r = (1/(2 * math.pi)) * math.sqrt((self.n**2 * self.W_r) / (2 * self.I))
        return self._v_r
    
    @staticmethod
    def moment_of_inertia(m, d):
        """
        Calculate moment of inertia for an atom about an axis.
        
        Args:
            m: Mass of atom (in amu)
            d: Distance from axis of rotation (in Angstroms)
        
        Returns:
            Moment of inertia (in kg*m^2)
        """
        m_kg = m * HinderedRotorPartitionFunction.AMU_TO_KG
        d_m = d * HinderedRotorPartitionFunction.ANGSTROM_TO_METER
        return m_kg * d_m**2
    
    @staticmethod
    def reduced_moment_of_inertia(atoms_dict):
        """
        Calculate reduced moment of inertia from dictionary of atoms.
        The reduced moment of inertia is the sum of the moments of inertia 
        of each atom in the adsorbate about the axis of rotation.
        
        Args:
            atoms_dict: Dict with atom names as keys and [mass (amu), distance (Angstrom)] as values
                       Example: {'H1': [1, 1.09], 'H2': [1, 1.09], ...}
        
        Returns:
            Reduced moment of inertia (in amu*Angstrom^2)
        """
        total = sum(m * d**2 for m, d in atoms_dict.values())
        return total
    
    @property
    def r_r(self):
        """
        Ratio of energy barrier height to harmonic oscillator frequency times Planck's constant.
        
        Returns:
            Dimensionless ratio r_r
        """
        if self._r_r is None:
            self._r_r = self.W_r / (self.H * self.vibrational_frequency)
        return self._r_r
    
    @property
    def T_r(self):
        """
        Dimensionless temperature T_r.
        
        Returns:
            Dimensionless temperature
        """
        if self._T_r is None:
            self._T_r = (self.K * self.T) / (self.H * self.vibrational_frequency)
        return self._T_r
    
    def q_HO_r(self):
        """
        Quantum harmonic oscillator partition function.
        
        Returns:
            Harmonic oscillator partition function
        """
        return math.exp(-1/(2 * self.T_r)) / (1 - math.exp(-1/self.T_r))
    
    @staticmethod
    def I_0(x):
        """
        Zeroth-order modified Bessel function of the first kind, I_0(x).
        Uses series expansion approximation.
        
        Args:
            x: Argument
        
        Returns:
            I_0(x)
        """
        return 1 + (x**2)/4 + (x**4)/64 + (x**6)/2304 + (x**8)/147456 + (x**10)/14745600
    
    def P_rot(self):
        """
        Rotational probability factor.
        
        Returns:
            Rotation probability factor
        """
        return math.sqrt(math.pi * self.r_r / self.T_r) * \
               math.exp(-self.r_r / (2 * self.T_r)) * \
               self.I_0(self.r_r / (2 * self.T_r))
    
    def f_rot(self):
        """
        Normalized rotational partition function with zero-point energy correction.
        
        Returns:
            Normalized partition function
        """
        return self.P_rot() * math.exp(1 / ((2 + 16 * self.r_r) * self.T_r))
    
    def q_rot(self):
        """
        Total rotational partition function.
        For asymmetric rotors, uses full partition function.
        For symmetric rotors, divides by symmetry number.
        
        Returns:
            Rotational partition function
        """
        if self.rotor_asymmetric:
            return self.f_rot() * self.q_HO_r()
        else:
            return self.f_rot() * self.q_HO_r() / self.symmetric_number



# Example usage
if __name__ == "__main__":
    # Constants
    ANGSTROM_TO_METER = 1e-10
    
    # Example: Hindered Translation
    trans = HinderedTranslationPartitionFunction(
        m=1.0,  # mass in amu
        W_x=0.1,  # barrier height in eV
        W_y=0.1,  # barrier height in eV
        b=3.92 * ANGSTROM_TO_METER,  # lattice constant in meters
        M=10,  # number of surface sites
        T=300  # temperature in K
    )
    
    print("Hindered Translation Results:")
    print(f"Vibrational frequency: {trans.vibrational_frequency:.2e} Hz")
    print(f"r_x: {trans.r_x:.4f}")
    print(f"T_x: {trans.T_x:.4f}")
    print(f"q_trans: {trans.q_trans():.4e}")
    print(f"f_trans: {trans.f_trans():.4e}")
    print()
    
    # Example: Hindered Rotor
    atoms_dict = {'H1': [1, 1.09], 'H2': [1, 1.09], 'H3': [1, 1.09], 'H4': [1, 1.09]}
    reduced_moi = HinderedRotorPartitionFunction.reduced_moment_of_inertia(atoms_dict)
    
    rotor = HinderedRotorPartitionFunction(
        W_r=0.1,  # barrier height in eV
        n=1,  # number of equivalent minima
        I=reduced_moi,  # reduced moment of inertia in amu*Angstrom^2
        T=300,  # temperature in K
        symmetric_number=2,  # for homonuclear diatomic
        rotor_asymmetric=True
    )
    
    print("Hindered Rotor Results:")
    print(f"Reduced moment of inertia: {reduced_moi:.2f} amu*Angstrom^2")
    print(f"Vibrational frequency: {rotor.vibrational_frequency:.2e} Hz")
    print(f"r_r: {rotor.r_r:.4f}")
    print(f"T_r: {rotor.T_r:.4f}")
    print(f"q_rot: {rotor.q_rot():.4e}")
    print(f"P_rot: {rotor.P_rot():.4e}")