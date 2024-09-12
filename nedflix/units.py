import numpy as np

class Units():

    def __init__(self):
        ## PHYSICS CONSTANTS
        #===========================================================================
        # NAME
        #===========================================================================

        self.name = "STD"                    # Default values
        self.linestyle = "solid"             # Default linestyle in plots
        self.markerstyle = "*"               # Default marker style
        self.colorstyle = "red"              # Default color style
        self.savefilename = "output.dat"     # Default color style

        #===============================================================================
        # ## MATH
        #===============================================================================
        self.PI=3.14159265                  # Pi
        self.PIby2=1.5707963268             # Pi/2
        self.sqr2=1.4142135624              # Sqrt[2]
        self.ln2 = np.log(2.0)

        #===============================================================================
        # ## EARTH
        #===============================================================================
        self.EARTHRADIUS = 6371.0           # [km] Earth radius
        #===============================================================================
        # ## SUN
        #===============================================================================
        self.SUNRADIUS = 109*self.EARTHRADIUS     # [km] Sun radius

        #===============================================================================
        # # PHYSICAL CONSTANTS
        #===============================================================================
        self.GF = 1.16639e-23               # [eV^-2] Fermi Constant
        self.Na = 6.0221415e+23                 # [mol cm^-3] Avogadro Number
        self.sw_sq = 0.2312                  # [dimensionless] sin(th_weinberg) ^2
        self.G  = 6.67300e-11                # [m^3 kg^-1 s^-2]
        self.alpha = 1.0/137.0               # [dimensionless] fine-structure constant

        #===============================================================================
        # ## UNIT CONVERSION FACTORS
        #===============================================================================
        # Energy
        self.TeV = 1.0e12                    # [eV/TeV]
        self.GeV = 1.0e9                     # [eV/GeV]
        self.MeV = 1.0e6                     # [eV/MeV]
        self.keV = 1.0e3                     # [eV/keV]
        self.Joule = 1/1.60225e-19           # [eV/J]
        # Mass
        self.kg = 5.62e35                    # [eV/kg]
        self.gr = 1e-3*self.kg               # [eV/g]
        # Time
        self.sec = 1.523e15                  # [eV^-1/s]
        self.hour = 3600.0*self.sec          # [eV^-1/h]
        self.day = 24.0*self.hour            # [eV^-1/d]
        self.year = 365.0*self.day           # [eV^-1/yr]
        self.yearstosec = self.sec/self.year # [s/yr]
        # Distance
        self.meter = 806554.815355           # [eV^-1/m]
        self.cm = 1.0e-2*self.meter          # [eV^-1/cm]
        self.km = 1.0e3*self.meter           # [eV^-1/km]
        self.fermi = 1.0e-15*self.meter      # [eV^-1/fm]
        self.angstrom = 1.0e-10*self.meter   # [eV^-1/A]
        self.AU = 149.60e9*self.meter        # [eV^-1/AU]
        self.parsec = 3.08568025e16*self.meter# [eV^-1/parsec]
        # Integrated Luminocity # review
        self.picobarn = 1.0e-36*self.cm**2   # [eV^-2/pb]
        self.femtobarn = 1.0e-39*self.cm**2  # [eV^-2/fb]
        # Presure
        self.Pascal = self.Joule/self.meter**3 # [eV^4/Pa]
        self.hPascal = 100.0*self.Pascal     # [eV^4/hPa]
        self.atm = 101325.0*self.Pascal      # [eV^4/atm]
        self.psi = 6893.0*self.Pascal        # [eV^4/psi]
        # Temperature
        self.kelvin = 1/1.1604505e4          # [eV/K]
        # Angle
        self.degree = self.PI/180.0          # [rad/degree]
        # magnetic field
        self.T = 0.000692445                 # [eV^2/T]

        # old notation
        self.cm3toev3 = 7.68351405e-15       # cm^3-> ev^3
        self.KmtoEv =5.0677288532e+9         # km -> eV
        self.yearstosec = 31536.0e3          # years -> sec

units = Units()
