#   Zachary Levine     26 March 2021 - 28 May 2021

# Translation of CircuitFaker from Mathematica  circuitFakerXX.nb
# Adapted from cfEntroPy04.py

# This version writes a single file
# For an even size problem only
# Problem size is 2X seed size in each dimension

# nZSeed - number of seed layers in Z  (also Y and X)
# nSeed - number of wiring seeds
# nZ - number of circuit layers in Z (also Y and X)
# pW - probabilty of getting a wiring seed
# pZ - probabilty of extending wire in Z direction (also Y and X)
# nCircuit - number of circuits to write
# nCol - number of columns for output images
# nDig - number of digits for serial number in file output
# showSeed - True to display the seeds
# showCircuit - True to display the circuit

# Programmer's note:  using np.random.seed to set the global random number
# generator.  This is not suitable for advanced applications.
# https://albertcthomas.github.io/good-practices-random-number-generators/

# 64x64x8 targeting Superbee

#################################### User set parameters
randomSeed = 20210506
nZSeed =  4
nYSeed =  8
nXSeed =  8
pW = 0.75
pZ = 0.5
pY = 0.80
pX = 0.80
nCircuit = 1e8#2000
nCol = 4
nDig = 5
doCMPfill    = False
showSeed     = False
showCircuitA = False       # circuit
showCircuitB = False       # circuit with CMP fill
showCircuitC = False       # CMP fill alone
subFigInch = 2.0           # size of each subfigure
#################################### end parameters

#################################### end functions
print('circuitFaker:  starts')
print('circuitFaker:  nZSeed nYSeed nXSeed', nZSeed, nYSeed, nXSeed)
print('circuitFaker:  pW pZ pY pX', pW, pZ, pY, pX)
print('circuitFaker:  nCircuit', nCircuit)

#                                     imports
from libtiff import TIFF
import matplotlib.pyplot as plt
import numpy as np
if doCMPfill :
  import scipy.ndimage as ndi
import time

#                                     derived parameters
nSeed = nZSeed * nYSeed * nXSeed
nZ = 2 * nZSeed
nY = 2 * nYSeed
nX = 2 * nXSeed
nRowSeed = int(np.ceil(nZSeed/nCol))
nRow     = int(np.ceil(nZ/nCol))

#                                     initialization
t0 = time.perf_counter()
np.random.seed(randomSeed)

################################################ functions

def doubleBlock (arr) :
  dim = np.array(np.shape(arr))
  arrDbl = np.empty( 2 * dim, dtype=arr.dtype )
  arrDbl[0::2,0::2] = arr
  arrDbl[0::2,1::2] = arr
  arrDbl[1::2,0::2] = arr
  arrDbl[1::2,1::2] = arr
  return arrDbl

# Creats a 3D array of binary values

def getWireSeed (nZSeed, nYSeed, nXSeed, pBernoulli) :
  return np.random.binomial(1, pBernoulli, [nZSeed,nYSeed,nXSeed])

# Create an instance of a circuit-faker circuit
# wireSeed - array of 0's and 1's which denote the wire seed points
# dim - dimensions of wireSeed
# circuit - output with all wiring
# wireSeedMaskZ - tells if a wiring point should be considered in
#   the Z direction
# The triple for loop loops over all the wire seed points and
#   fills in wiring in the Z, Y, and X directions as needed
#   each wire seed is put at an even indexed point, and the
#   wire is at the next higher odd value (which always exists)
#   wires wrap for the purpose of creation, but they are put at
#   the last index value for each dimension.  A given wire layer
#   will be for Y or X but not both (hence the "if k % 2" statment).

def getCircuitFromSeed ( wireSeed, pZ, pY, pX ) :
  dim = np.array(np.shape(wireSeed))
  circuit = np.full( 2 * dim, 0 )
  wireSeedMaskZ = np.roll(wireSeed, 1, axis=0) * wireSeed
  wireSeedMaskY = np.roll(wireSeed, 1, axis=1) * wireSeed
  wireSeedMaskX = np.roll(wireSeed, 1, axis=2) * wireSeed
  for k in range(dim[0]) :
    for j in range(dim[1]) :
      for i in range(dim[2]) :
        circuit[2*k,2*j,2*i] = wireSeed[k,j,i]
        if wireSeedMaskZ[k,j,i] == 1 :
          circuit[2*k+1,2*j,2*i] = np.random.binomial(1, pZ)
        if k % 2 == 1 :
          if wireSeedMaskY[k,j,i] == 1 :
            circuit[2*k,2*j+1,2*i] = np.random.binomial(1, pY)
        else :
          if wireSeedMaskX[k,j,i] == 1 :
            circuit[2*k,2*j,2*i+1] = np.random.binomial(1, pX)
  return circuit

# cmpFill sets all the elements to 1 which will not change the 
#   connectivity of the circuit.  Uses + as Boolean OR and
#   1-x as Boolean NOT.

def cmpFill (circuit) :
  kernel = np.ones((3,3,3),np.uint8)
  cmpFil = 1 - ndi.binary_dilation(circuit,
                    structure=kernel).astype(circuit.dtype)
  return circuit + cmpFil

def writeReconstrIdl ( circuit, i, nDig ) :
# next line a prototype ... delete soon
#  fileID = open('reconstr.idl.' + ( '%.5d' % i ), 'w' )
  fileID = open('reconstr.idl.'
             + ((( (( '%.' + ( '%d' % nDig ) + 'd' )) % i ))), 'w' )
  np.savetxt(fileID, ['20000524  fmtGot'], fmt='%s')
  np.savetxt(fileID, [[2*nXSeed, 2*nYSeed, 2*nZSeed]], fmt='4 1 %d %d %d 5')
  np.savetxt(fileID, circuit.flatten(), fmt='%d')
  fileID.close()

######################################### Main executables

# Fig. 1 is image of wiring seeds
# Fig. 2 is image of circuit
# Fig. 3 is image of circuit with CMP fill
# Fig. 4 is image of CMP fill
from tqdm import tqdm

for iC in tqdm(np.arange(nCircuit)) :
  wireSeed = getWireSeed (nZSeed, nYSeed, nXSeed, pW)

  circuit = getCircuitFromSeed ( wireSeed, pZ, pY, pX )

  if showSeed :
    plt.figure(1,figsize=(subFigInch*nCol,subFigInch*nRowSeed))
    for k in np.arange(nZSeed) :
      plt.subplot( nRowSeed, nCol, k+1 )
      plt.imshow(1-wireSeed[k])
      plt.axis('off')
      plt.gray()
    plt.tight_layout()
    plt.show(block=False)
#
  if showCircuitA :
    plt.figure(2,figsize=(subFigInch*nCol,subFigInch*nRow))
    for k in np.arange(nZ) :
      plt.subplot( nRow, nCol, k+1 )
      plt.imshow(1-circuit[k])
      plt.axis('off')
      plt.gray()
    plt.tight_layout()
    plt.show(block=False)
#
  if doCMPfill :
    circuitB = cmpFill (circuit)
    if showCircuitB :
      plt.figure(3,figsize=(subFigInch*nCol,subFigInch*nRow))
      for k in np.arange(nZ) :
        plt.subplot( int(np.ceil(nZ/nCol)), nCol, k+1 )
        plt.imshow(1-circuitB[k])
        plt.axis('off')
        plt.gray()
      plt.tight_layout()
      plt.show(block=False)
    if showCircuitC :
      plt.figure(4,figsize=(subFigInch*nCol,subFigInch*nRow))
      for k in np.arange(nZ) :
        plt.subplot( int(np.ceil(nZ/nCol)), nCol, k+1 )
        plt.imshow(1-(circuitB[k]-circuit[k]))
        plt.axis('off')
        plt.gray()
      plt.tight_layout()
      plt.show(block=False)
#
  if doCMPfill :
    writeReconstrIdl( circuitB, iC, nDig )
  #else :
  #  writeReconstrIdl( circuit, iC, nDig )
# end of main loop

print('circuitFaker: %.6f sec' % (time.perf_counter()-t0) )
