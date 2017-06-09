import numpy as np
import argparse
FLAGS = None
SPINDIM = 3

def spin_operators():
#  print("Prepare spin operators")
  X = 1./np.sqrt(2.)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
  Y = 1./np.sqrt(2.)*np.array([[0.,-1.0j,0.],[1.0j,0.,-1.0j],[0.,1.0j,0.]])
  Z = np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]])

  sx_list = []
  sy_list = []
  sz_list = []

  L=FLAGS.spin_number
  for i_site in range(L):
    if i_site==0:
      sx = X
      sy = Y
      sz = Z
    else:
      sx = np.eye(SPINDIM)
      sy = np.eye(SPINDIM)
      sz = np.eye(SPINDIM)

    for j_site in range(1,L):
      if j_site==i_site:
        sx = np.kron(sx,X)
        sy = np.kron(sy,Y)
        sz = np.kron(sz,Z)
      else:
        sx = np.kron(sx,np.eye(SPINDIM))
        sy = np.kron(sy,np.eye(SPINDIM))
        sz = np.kron(sz,np.eye(SPINDIM))

    sx_list.append(sx)
    sy_list.append(sy)
    sz_list.append(sz)

  return sx_list,sy_list,sz_list

def bilinear(sx_list,sy_list,sz_list):
#  print('Bilinear part...')
  L=FLAGS.spin_number

  dim = SPINDIM**L
  ham = np.reshape(np.zeros(dim*dim),(dim,dim))

  for i in range(L):
    j=np.mod(i+1,L)
    ham=ham+np.tensordot(sz_list[i],sz_list[j],axes=(1,0))
    ham=ham+np.tensordot(sx_list[i],sx_list[j],axes=(1,0))
    ham=ham+np.tensordot(sy_list[i],sy_list[j],axes=(1,0))

  return ham

def biquadratic(sx_list,sy_list,sz_list):
#  print('Biquadratic part...')
  L=FLAGS.spin_number

  dim = SPINDIM**L
  ham = np.reshape(np.zeros(dim*dim),(dim,dim))

  for i in range(L):
    j=np.mod(i+1,L)
    hamL=np.tensordot(sz_list[i],sz_list[j],axes=(1,0))
    hamL=hamL+np.tensordot(sx_list[i],sx_list[j],axes=(1,0))
    hamL=hamL+np.tensordot(sy_list[i],sy_list[j],axes=(1,0))
    ham=ham+np.tensordot(hamL,hamL,axes=(1,0))

  return ham

def entanglementSpectrumV(psi):
    X,Y,Z = np.linalg.svd(psi)
    S=Y**2/sum(Y**2)
    es = -np.log(S)
    ee = sum(es*S)
    eg=X[:,0]
    return (ee, es, eg)

def printSpectrumOneliner(es):
    for e in es:
        print(e, end=" ")
    print()
    return

def main():
  L = FLAGS.spin_number
  La = FLAGS.spin_number_A
  if La > L:
    La = L//2
  Lb = L-La
  dimA = SPINDIM**La
  dimB = SPINDIM**Lb

#  Jl=FLAGS.J_bilinear
#  Jq=FLAGS.J_biquadratic
  sx_list,sy_list,sz_list = spin_operators()
  hamL = bilinear(sx_list,sy_list,sz_list)
  hamQ = biquadratic(sx_list,sy_list,sz_list)

  n_theta = 362880
#  n_theta = 120
  for theta_i in range(n_theta):
    theta = 2.*np.pi/n_theta*theta_i
    Jl = np.cos(theta)
    Jq = np.sin(theta)

    ham = Jl*hamL + Jq*hamQ
    w,v = np.linalg.eigh(ham)
    print(theta_i,end=' ')
#    printSpectrumOneliner(w)
    gs = np.reshape(v[:,0],(dimA,dimB))
    ee,es,eg = entanglementSpectrumV(gs)
    printSpectrumOneliner(es)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
            description='Bilinear-Biquadratic model: Entanglement spectrum.',
            epilog='End of description.')
  parser.add_argument('--spin_number', type=int, default=6,
                      help='number of spins (default 6)')
  parser.add_argument('--spin_number_A', type=int, default=3,
                      help='number of spins in subsystem A (default 3)')
  parser.add_argument('--J_bilinear', type=float, default=3.0,
                      help='bilinear coupling strength (default 3.0)')
  parser.add_argument('--J_biquadratic', type=float, default=1.0,
                      help='biquadratic coupling strength (default 1.0)')
  FLAGS = parser.parse_args()
  main()
