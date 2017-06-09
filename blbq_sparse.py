import scipy.sparse as sparse
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import argparse
FLAGS = None
SPINDIM = 3

def spin_operators():
  X = sparse.csr_matrix(1./np.sqrt(2.)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]))
  Y = sparse.csr_matrix(1./np.sqrt(2.)*np.array([[0.,-1.0j,0.],[1.0j,0.,-1.0j],[0.,1.0j,0.]]))
  Z = sparse.csr_matrix(np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]]))

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
      sx = sparse.csr_matrix(np.eye(SPINDIM))
      sy = sparse.csr_matrix(np.eye(SPINDIM))
      sz = sparse.csr_matrix(np.eye(SPINDIM))

    for j_site in range(1,L):
      if j_site==i_site:
        sx = sparse.kron(sx,X,'csr')
        sy = sparse.kron(sy,Y,'csr')
        sz = sparse.kron(sz,Z,'csr')
      else:
        sx = sparse.kron(sx,np.eye(SPINDIM),'csr')
        sy = sparse.kron(sy,np.eye(SPINDIM),'csr')
        sz = sparse.kron(sz,np.eye(SPINDIM),'csr')

    sx_list.append(sx)
    sy_list.append(sy)
    sz_list.append(sz)

  return sx_list,sy_list,sz_list

def bilinear(sx_list,sy_list,sz_list):
#  print('Bilinear part...')
  L=FLAGS.spin_number

  dim=SPINDIM**L
  ham=sparse.csr_matrix((dim,dim))

  for i in range(L):
    j=np.mod(i+1,L)
    ham=ham+sz_list[i]*sz_list[j]
    ham=ham+sx_list[i]*sx_list[j]
    ham=ham+sy_list[i]*sy_list[j]

  return ham

def biquadratic(sx_list,sy_list,sz_list):
#  print('Biquadratic part...')
  L=FLAGS.spin_number

  dim = SPINDIM**L
  ham=sparse.csr_matrix((dim,dim))

  for i in range(L):
    j=np.mod(i+1,L)
    hamL=sz_list[i]*sz_list[j]
    hamL=hamL+sx_list[i]*sx_list[j]
    hamL=hamL+sy_list[i]*sy_list[j]
    ham=ham+hamL*hamL

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

  n_theta = 800000
#  n_theta = 120
  for theta_i in range(n_theta):
    theta = 2.*np.pi/n_theta*theta_i
    Jl = np.cos(theta)
    Jq = np.sin(theta)

    ham = Jl*hamL + Jq*hamQ
    try:
      w,v = arp.eigsh(ham,k=1,which='SA',return_eigenvectors=True)
      print(theta_i,end=' ')
#      printSpectrumOneliner(w)
      gs = np.reshape(v[:,0],(dimA,dimB))
      ee,es,eg = entanglementSpectrumV(gs)
      printSpectrumOneliner(es)
    except:
      print('#',theta_i,'No convergence')

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
