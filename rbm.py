#******************************************************************************
#
# PROGRAM NAME : rbm.py -> RBM [ Reduced Basis Method ]
#
# AUTHOR : Josh Belieu <fletch>
#
# DATE CREATED : 01.10.24
#
# PURPOSE : Use RBM to to build an emulator of porvided data and compare
#           against sufficiently accurate data.
#
# The sufficiently accurate data in this project is synthesized data of a
# hydrogen atom in the presence of a yukawa potential.
#
# Special thanks is needed for eveeryone who has helped me through this
# project and provided access to absorb their code into my project. So, thank
# you to,
#
# Pablo Giuliani, PhD
# Megan Campbell
# Kyle Godbey, PhD
#
# ~ and ~
#
# Members of the ASCSN network.
#
#******************************************************************************

#==============================================================================
"                               Begin Program                                 "
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
"                             Program Parameters                              "
#------------------------------------------------------------------------------

"""
catplot settings and filename
"""

cat_plot_flag    = 0
cat_plot_save    = 0
catplot_filename = 'catplot.png'

"""
alpha vs eigenenergy settings and filename
"""

alpha_vs_eigenenergy_plot_flag = 0
alpha_vs_ee_plot_save          = 0
alpha_vs_ee_filename           = 'a.vs.ee.plot.png'

"""
super plot settings and filename
"""

super_plot_flag     = 0
super_plot_save     = 0
super_plot_filename = 'super.plot.png'

"""
super plot settings and filename
"""

rbm_basis_plot_flag = 0
rbm_basis_plot_save = 0

# the plot names must sadly be hard coded. they make
# use of a loop in the code. see line 558


general_plot_flag = 1 # leave this on if you want to see plots.


all_plot = 0 # turn on if you want to plot everything!
all_save = 0 # turn on if you want to save everything!

if all_plot == 1 :

    cat_plot_flag                  = 1
    alpha_vs_eigenenergy_plot_flag = 1
    rbm_basis_plot_flag            = 1
    super_plot_flag                = 1

if all_save == 1 :

    cat_plot_save         = 1
    alpha_vs_ee_plot_save = 1
    rbm_basis_plot_save   = 1
    super_plot_save       = 1

eigenenergy_table_print_flag = 0 # system data table.


"""
filenames that load in data.
"""

afile = 'alphas.txt' # alphas used to generate wave functions.
hf_file = 'hf.dat' # file containing pertinant high fidelity data.

"""
misc. parameters used in the code.
"""

biggest_x  = 110. # largest modeled distance
smallest_x = 1e-5 # smallest modeled distance

sli = 100 # an integer to SLIce out a certain
          # number of rows in a matrix.

pad = 12 # fstring pad, used in printed terminal.

lower_comp  = 2 # smallest number of components to loop through
                # MUST be >= 2.
higher_comp = 3 # biggest number of components to loop through. odd
               # data when apporaching number of alphas used.
step_comp   = 1 # interval of increase between adjacent components.

#------------------------------------------------------------------------------
"                             Imported Libraries                              "
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time as tm

#------------------------------------------------------------------------------
"                                 RBM class                                   "
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# PURPOSE : Use provided wavefunctions in conjucntion with a desired step size
#           and number of components to build a stand-in wavefunction basis.
#           Then, reconstruct the desired hamiltonian in order to begin
#           passing the necessary parameter(s) and therefore calculate the
#           eigenenergies of the system.
#           
# INPUTS :
# ------
#
# h : Float. The desired step size of the system. **NOTE** h is used to 
#     construct an array (self.x) which has an effect on the dimensions of the
#     matrices used to form the fundamental equation of QM which MUST have 
#     similar dimensions to m - the passed wavefunctions.
#
# m : Array-like. A collection of wavfunctions ordered column-wise into a
#     matrix. I use vstack to accomplish this.
#
# components : Integer. The desired number of comonents we use to expand a set
#              of reproduced wavefunctions.
#
# OUTPUTS :
# -------
#
# Access what you will but typically we want to access the eigenenergies of
# the emulated system. This is done through the RBM_solve routine within the
# class.
#
#------------------------------------------------------------------------------

class RBM :

    def __init__ ( self , h , m , components ) :

        #
        ## Input parameters.
        #

        self.m = m
        self.h = h
        self.components = components

        #
        ## Use SVD to obtain a reduced set of wavefunctions
        ## and create a copy of it.
        #

        self.psi = np.array ( self.getReducedBasis() )
        self.phi = self.psi

        #
        ## Initialize an array to model the physical distance
        ## of the system. This intialization should match
        ## the provided data as best you can.
        #

        self.x_max = biggest_x
        self.x_min = smallest_x

        self.x = np.arange ( self.x_min , self.x_max , h )

        #
        ## Second derivative matrix via forward difference method.
        #

        self.d2_forw = self.forward_second_derivative_matrix()

        #
        ## Second derivative matrix via backward difference method.
        #

        self.d2_back = self.backward_second_derivative_matrix()

        #
        ## Second derivative matrix via central difference method.
        #

        self.d2_cent = self.central_second_derivative_matrix()

        #
        ## Use the previous difference methods to construct a total
        ## second derivative matrix. In this combination I like to 
        ## refer to it as the mixed or stitched difference method.
        #

        self.d2 = self.second_derivative_matrix()

        #
        ## A matrix representing the Coulomb portion of
        ## the potential interaction.
        #

        self.pot_coul = self.potential_matrix_coul() 

        #
        ## A matrix representing the Yukawa portion of
        ## the potential interaction.
        #

        self.pot_yuk = self.potential_matrix_yuk()

        #
        ## Array constructors.
        #

        self.compvec = np.zeros(components)
        self.array = self.create_array()

        #
        ## Consider the Hamiltonian represented as :
        ## H = M_0 + a * M_1
        ## Below are the initializations for those matrices.
        ## a is a scalar.
        #

        self.M_0 = np.array(self.array)
        self.M_1 = np.array(self.array)

        #
        ## This is used if the wavefunction basis is not
        ## orthogonal. I don't use it - Fletch.
        #

        self.N = np.array(self.array)

        #
        ## Below are the constructors for M_0 and M_1.
        ## @ represents matrix multipliation. Notice
        ## that the matrices are real symmetric.
        #

        for i in range(self.components):
            for j in range(i, self.components):
                self.M_0[i][j] = self.phi[i] @ (self.d2+self.pot_coul) @ self.psi[j]
                self.M_0[j][i] = self.M_0[i][j]
                self.M_1[i][j] = self.phi[i] @ self.pot_yuk @ self.psi[j]
                self.M_1[j][i] = self.M_1[i][j]
        
        #
        ## Again - I don't use this but we still
        ## construct it, just in case...
        #

        self.create_N()

    #
    ## Construct a square matrix of zeroes of
    ## dimension given by components. 
    #

    def create_array ( self ) :

        array = []

        for i in range ( self.components ) :

            array.append ( self.compvec )

        return array
    
    #
    ## Using central difference method, construct
    ## a second derivative matrix.
    #
    
    def central_second_derivative_matrix ( self ) :

        N = len ( self.x ) # length of x array
        dx = self.h # desired step size

        main_diag = np.ones ( N ) * ( -205. / 72. / dx ** 2 )
        off_diag = np.ones ( N - 1 ) * ( 8. / 5. / dx ** 2 )
        off_diag2 = np.ones ( N - 2 ) * ( -1. / ( 5. * dx ** 2 ) )
        off_diag3 = np.ones ( N - 3 ) * (8. / 315. / dx ** 2 )
        off_diag4 = np.ones ( N - 4 ) * ( -1. / 560. / dx ** 2 )

        D2 = np.diag ( main_diag ) + np.diag ( off_diag , k = 1 ) + np.diag ( off_diag , k = -1 ) + \
             np.diag (off_diag2 , k = 2 ) + np.diag ( off_diag2 , k = -2 ) + np.diag ( off_diag3 , k = 3 ) + \
             np.diag ( off_diag3 , k = -3 ) + np.diag ( off_diag4 , k = -4 ) + np.diag ( off_diag4 , k = 4 )

        return D2
    
    #
    ## Using forward difference method, construct
    ## a second derivative matrix.
    #
    
    def forward_second_derivative_matrix( self ):

        N = len ( self.x ) # length of x array
        dx = self.h # desired step size

        main_diag = np.ones ( N ) * ( 29531. / 5040. / dx ** 2 )
        off_diag = np.ones ( N - 1 ) * -962. / 35. / dx ** 2
        off_diag2 = np.ones ( N - 2 ) * ( 621. / 10. / dx ** 2 )
        off_diag3 = np.ones ( N - 3 ) * ( -4006. / 45. / dx ** 2 )
        off_diag4 = np.ones ( N - 4 ) * ( 691. / 8. / dx ** 2 )
        off_diag5 = np.ones ( N - 5 ) * ( -282. / 5. / dx ** 2 )
        off_diag6 = np.ones ( N - 6 ) * ( 2143. / 90. / dx ** 2 )
        off_diag7 = np.ones ( N - 7 ) * ( -206. / 35. / dx ** 2 )
        off_diag8 = np.ones ( N - 8 ) * ( 363. / 560. / dx ** 2 )

        D2 = np.diag ( main_diag ) + np.diag ( off_diag , k = 1 ) + np.diag ( off_diag2 , k = 2 ) + \
             np.diag ( off_diag3 , k = 3 ) + np.diag ( off_diag4 , k = 4 ) + np.diag ( off_diag5 , k = 5 ) +  \
             np.diag ( off_diag6 , k = 6 ) + np.diag ( off_diag7 , k = 7 ) + np.diag ( off_diag8 , k = 8 )

        return D2
    
    #
    ## Using backward difference method, construct
    ## a second derivative matrix.
    #
    
    def backward_second_derivative_matrix( self ):

        N = len ( self.x ) # length of x array
        dx = self.h # desired step size

        main_diag = np.ones ( N ) * ( 29531. / 5040. / dx ** 2 )
        off_diag = np.ones ( N - 1 ) * -962. / 35. / dx ** 2
        off_diag2 = np.ones ( N - 2 ) * ( 621. / 10. / dx ** 2 )
        off_diag3 = np.ones ( N - 3 ) * ( -4006. / 45. / dx ** 2 )
        off_diag4 = np.ones ( N - 4 ) * ( 691. / 8. / dx ** 2 )
        off_diag5 = np.ones ( N - 5 ) * ( -282. / 5. / dx ** 2 )
        off_diag6 = np.ones ( N - 6 ) * ( 2143. / 90. / dx ** 2 ) 
        off_diag7 = np.ones ( N - 7 ) * ( -206. / 35. / dx ** 2 )
        off_diag8 = np.ones ( N - 8 ) * ( 363. / 560. / dx ** 2 )

        D2 = np.diag ( main_diag ) + np.diag ( off_diag , k = -1 ) + np.diag ( off_diag2 , k = -2 ) + \
             np.diag ( off_diag3 , k = -3 ) + np.diag ( off_diag4 , k = -4 ) + np.diag ( off_diag5 , k= -5 ) +  \
             np.diag ( off_diag6 , k = -6 ) + np.diag ( off_diag7 , k = -7 ) + np.diag ( off_diag8 , k = -8 )

        return D2
    
    #
    ## Combine (mix/stitch) all previous derivative matrices
    ## and return the total derivative matrix.
    #

    def second_derivative_matrix ( self ) :

        cen_d2_mat = self.d2_cent
        for_d2_mat = self.d2_forw
        bac_d2_mat = self.d2_back

        for_d2_mat[sli:]=0 # remove all entries of forward
                           # matrix from row 100 on.
        cen_d2_mat[0:sli]=0 # remove all entries of central matrix
                            # from beginning to row 100.

        bac_d2_mat[:-sli] = 0 # remove all entries of backwards matrix
                              # from end to row -100 (100 from end).
        cen_d2_mat[-sli:]=0 # remove all entries of central
                            # matrix from row -100 to end.


        #
        ## add 'em up!
        #

        D2 = cen_d2_mat + for_d2_mat + bac_d2_mat

        return -0.5 * D2 # -1/2 for fund. QM eq.
    
    #
    ## diagonal matrix of Coulomb potential.
    #

    def potential_matrix_coul ( self ) :

        return np.diag ( 1. / self.x )
    
    #
    ## diagonal matrix of Yukawa potential.
    #
    
    def potential_matrix_yuk ( self ) :

        return np.diag ( np.exp ( -self.x ) / self.x )
    
    #
    ## use SVD on provided wavefunctions to obtain a reduced
    ## basis. Truncate basis to desired number of components.
    #

    def getReducedBasis ( self ) :

        U , sigma , Vh = np.linalg.svd ( self.m )

        reduced_basis = Vh [ : self.components ]

        return reduced_basis
    
    #
    ## Below we create the four matrices that we need, using np.dot
    ## to get the dot product between the phi and psi list.
    #

    def M0 ( self , i , j ) :

        M0 = np.dot ( self.psi [ j ] , np.dot ( self.d2 + self.pot_coul , self.phi [ i ] ) )

        return M0

    def M1 ( self , i , j ) :

        M1 = np.dot ( self.psi [ j ] , np.dot ( self.pot_yuk , self.phi [ i ] ) )

        return M1
    
    #
    ## Note that this function takes in the alpha parameter.
    #

    def create_H_hat ( self , alpha ) :

        H_hat = self.M_0 + alpha * self.M_1

        return H_hat
    
    def create_N ( self ) :

        for i in range ( self.components ) :
            for j in range ( i , self.components ) :

                self.N [ i , j ] = self.phi [ i ] @ self.psi [ j ]
                self.N [ j , i ] = self.N [ i , j ]

    #
    ## Note that this function takes in the alpha parameter to
    ## pass it on to the create_H_hat() function.
    #

    def RBM_solve ( self , alpha ) :

        H_hat = self.create_H_hat ( alpha )

        #self.create_N()

        #evals, evects = np.linalg.eigvalsh(H_hat)
        # evals = np.linalg.eigvalsh(H_hat)

        evals, eigvecs = np.linalg.eigh ( H_hat )

        return evals #, evects

#------------------------------------------------------------------------------
"                              Main Body of Code                              "
#------------------------------------------------------------------------------

alphasl = np.loadtxt ( afile , dtype = float ) # list of alphas.

wfl = [] # Wave Function List.

#
## Load in every wavefunction of each alpha. Normalize wavefunction,
## and make sure the wavefunction is properly oriented.
#

for alphav in alphasl :

    file = 'wf.' + str ( alphav ) + '.dat'

    cwf = -1 * np.loadtxt ( file , dtype = float ) [ :: -1 ] # reflect over
                                                             # x and y.

    nwf = cwf / np.linalg.norm ( cwf ) # normalize.

    wfl.append ( nwf ) # add to wfl.

#
## stack the normalized wave functions column wise.
#

wfm = np.vstack ( tuple ( wfl ) )

h = np.abs( biggest_x - smallest_x ) / len ( wfm [ 0 ] )

#
## load in high fidelity data.
#

hfalphas , hfee = np.loadtxt ( hf_file , dtype = float , unpack = True )

#
## Setup cat plot.
#

if cat_plot_flag == 1 :

    catFig , catAx = plt.subplots()

pfmt = f"|{"RBM Basis":^{pad}}|{"Str. Par.":^{pad}}|{"HF Energy":^{pad}}|{"RBM Energy":^{pad}}|{"Rel. Error":^{pad}}|"
wfmt = "|" + "-" * pad

if eigenenergy_table_print_flag == 1 :

    print ("=" * len (  pfmt ) )
    print ( pfmt )
    print ( wfmt * 5 + '|' )

mel = [] # Mega Energy List

for comp in range ( lower_comp , higher_comp , step_comp ) :
  
  times = []
  errors = []
  el = [] # Energy List.

  RBM_instance = RBM ( h , wfm , comp )

  if rbm_basis_plot_flag == 1 :
      
      px = np.arange ( smallest_x , biggest_x , h )
      
      plt.figure()
      
      for i in range ( comp ) :
          
          plt.plot ( px , -RBM_instance.getReducedBasis() [ i ] )

          plt.xlim ( 0 , 12 )

          plt.title ( 'RBM basis : ' + str ( comp ) )
          
          if rbm_basis_plot_save == 1 :
              
              plt.savefig ( 'rbm.plot.' + str ( comp ) + '.comp.png' )

  #
  ## for each alpha used in high fidelity data pass an alpha to our emulator
  ## and calculate the associated eigenenergy. Calculate the relative error
  ## compared to high fidelity energy and eventually plot it.
  #

  for j in range ( len ( hfalphas ) ) :

    alpha = hfalphas [ j ]

    # Instead of having a timing function, we do the timing
    # calculation outside of the class.

    timeDif = 0
    
    for i in range ( 10 ) :
      
      time1 = tm.time()

      #evals, evects = RBM_instance.RBM_solve(alpha)
      evals = RBM_instance.RBM_solve ( alpha ) 
      
      time2 = tm.time()

      timeDif += ( time2 -time1 )

    timeDif /= 10

    value = evals [ 0 ]

    el.append ( value )

    #
    ## Compare high fidelity data and our predicted energy.
    #

    errorDif = abs ( hfee [ j ] - value ) / hfee [ j ]

    fmt = f"|{str(comp).zfill(2):^{pad}}|{alpha:^{pad}.2f}|{hfee[j]:^{pad}.3f}|{value:^{pad}.3f}|{errorDif:^{pad}.3f}|"

    if eigenenergy_table_print_flag == 1 :
        
        print( fmt )

    times.append ( timeDif )
    errors.append ( errorDif )

  mel.append ( el )

  if cat_plot_flag == 1 :
      
      catAx.scatter ( times , errors , label = "RBM Basis =" + str ( comp ) )

if eigenenergy_table_print_flag == 1 :
    
    print ( "=" * len ( pfmt ) )

if cat_plot_flag == 1 :
    
    catAx.set ( xscale = 'log' , yscale = 'log' , 
            xlabel = 'Time (s)' , ylabel = 'Eigenvalue Relative Error' )
    #catAx.set(xlabel='Time (s)',ylabel='Eigenvalue Relative Error' )

    catAx.grid ( ls = '--' , alpha = 0.6 )
    catAx.legend()

    if cat_plot_save == 1 :

        plt.savefig ( catplot_filename )

if super_plot_flag == 1 :

    plt.figure()

    
    for wfv in wfm :
        
        plt.plot(wfv)

        plt.xlim(0,100)

    plt.title ( 'Super Plot' )
    plt.xlabel ( r'Distance [ ~0.5 $\AA$ ]' )
    plt.ylabel ( r'$\psi$' )

    plt.grid ( ls = '--' , alpha = 0.6 )

    if super_plot_save == 1 :

        plt.savefig ( super_plot_filename )

if alpha_vs_eigenenergy_plot_flag == 1 :
    
    plt.figure()

    for loe in mel :

        plt.scatter(hfalphas,loe)

    plt.scatter(hfalphas,hfee,color='g',label='HF')


    plt.legend()

    if alpha_vs_ee_plot_save == 1 :

        plt.savefig ( alpha_vs_ee_filename )

if general_plot_flag == 1 :
    
    plt.show()

#------------------------------------------------------------------------------
"                               End Program                                   "
#==============================================================================