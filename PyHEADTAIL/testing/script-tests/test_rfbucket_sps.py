from scipy.constants import c
from SPS import SPS


machine = SPS(machine_configuration='Q26-injection',
              n_segments=1, longitudinal_mode='non-linear')
lm = machine.longitudinal_map
bucket26 = lm.get_bucket(gamma=machine.gamma)

machine = SPS(machine_configuration='Q20-injection',
              n_segments=1, longitudinal_mode='non-linear')
lm = machine.longitudinal_map
bucket20 = lm.get_bucket(gamma=machine.gamma)

epsn_z = 0.35

print ("Q26 length: {:g}, Q20 length: {:g}".format(
    bucket26.bunchlength_single_particle(epsn_z),
    bucket20.bunchlength_single_particle(epsn_z)))

# print ("Q26 length: {:g}, Q20 length: {:g}".format(
#     bucket26.bunchlength_single_particle(epsn_z)/c*1e9*6,
#     bucket20.bunchlength_single_particle(epsn_z)/c*1e9*6))
