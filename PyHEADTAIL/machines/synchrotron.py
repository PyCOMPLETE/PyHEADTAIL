import numpy as np
from scipy.constants import c

from PyHEADTAIL.general.decorators import deprecated
from PyHEADTAIL.particles import generators
from PyHEADTAIL.general.element import Element
from PyHEADTAIL.trackers.rf_bucket import RFBucket
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap, RFSystems
from PyHEADTAIL.trackers.wrapper import LongWrapper
from PyHEADTAIL.particles.slicing import UniformBinSlicer


class Synchrotron(Element):
    def __init__(
        self,
        optics_mode,
        charge=None,
        mass=None,
        p0=None,
        circumference=None,
        n_segments=None,
        name=None,
        s=None,
        alpha_x=None,
        beta_x=None,
        D_x=None,
        alpha_y=None,
        beta_y=None,
        D_y=None,
        accQ_x=None,
        accQ_y=None,
        Qp_x=0,
        Qp_y=0,
        app_x=0,
        app_y=0,
        app_xy=0,
        longitudinal_mode=None,
        Q_s=None,
        alpha_mom_compaction=None,
        h_RF=None,
        V_RF=None,
        dphi_RF=None,
        p_increment=None,
        RF_at="middle",
        wrap_z=False,
        other_detuners=[],
        use_cython=False,
    ):
        """
        Creates a synchrotron.

        Parameters
        ----------
        optics_mode : 'smooth', 'non-smooth'

             - 'smooth': the optics parameters are the same for all segments;
             - 'non-smooth': the optics parameters are different for each segment.

        charge : C
            reference particle charge in Coulomb.

        mass : kg
            reference particle mass in Kg.

        p0 : kg m/s
            reference particle momentum.

        circumference : m
            ring circumference (to be provided only if optics_mode is 'smooth').

        n_segments :
            Number of segments in the machine (to be provided if
            optics_mode is 'smooth', otherwise it is inferred by the length of s).

        name :
            Name of the locations in between segments. The length of the array
            should be n_segments + 1.

        s : m, array
            Longitudinal positions at which the machine is cut
            in segments. The length of the array should be n_segments + 1.
            The last value in the array provides the ring circumference.

        alpha_x :
            Horizontal alpha twiss parameter at each segment
            (cannot be provided if optics_mode is 'smooth'). In this case,
            the length of the array should be n_segments + 1. The last point of
            the array should be equal to the first (periodic condition).

        beta_x : m
            Horizontal beta twiss parameter at each segment. It has
            to be a scalar if optics_mode is 'smooth', an array if optics_mode
            is 'non-smooth'. In this case, the length of the array should be
            n_segments + 1. The last point of the array should be equal to the
            first (periodic condition).

        D_x : m
            Horizontal beta twiss parameter at each segment. It has
            to be a scalar if optics_mode is 'smooth', an array if optics_mode
            is 'non-smooth'. In this case, the length of the array should be
            n_segments + 1. The last point of the array should be equal to the
            first (periodic condition).

        alpha_y :
            Vertical alpha twiss parameter at each segment
            (cannot be provided if optics_mode is 'smooth'). In this case,
            the length of the array should be n_segments + 1. The last point of
            the array should be equal to the first (periodic condition).

        beta_y : m
            Vertical beta twiss parameter at each segment. It has
            to be a scalar if optics_mode is 'smooth', an array if optics_mode
            is 'non-smooth'. In this case, the length of the array should be
            n_segments + 1. The last point of the array should be equal to the
            first (periodic condition).

        D_y : m
            Vertical beta twiss parameter at each segment. It has
            to be a scalar if optics_mode is 'smooth', an array if optics_mode
            is 'non-smooth'. In this case, the length of the array should be
            n_segments + 1. The last point of the array should be equal to the
            first (periodic condition).

        accQ_x :
            Horizontal tune or phase advance:

            - for 'optics_mode' = 'smooth' this is the horizontal tune;
            - for 'optics_mode' = 'non-smooth' this is the horizontal phase advance
              per segment in units of (2*pi). The length of the array should be
              n_segments + 1. The last point of the array should be the
              horizontal tune.

        accQ_y :
            Vertical tune or phase advance:

             - for 'optics_mode' = 'smooth' this is the vertical tune;
             - for 'optics_mode' = 'non-smooth' this is the vertical phase advance
               per segment in units of (2*pi). The length of the array should be
               n_segments + 1. The last point of the array should be the
               vertical tune.

        Qp_x,Qp_y :
            Horizontal and vertical chromaticity (dQ/dp), the detuning
            is shared over segments.

        app_x,app_y,app_xy :
            Amplitude detuning coefficients (anharmonicities).

        longitudinal_mode : 'linear', 'non-linear'
            Longitudinal mode:
             - 'linear': linear longitudinal force (RF cavity)
             - 'non-linear': sinusoidal longitudinal force (RF cavities). Multiple
                harmonics can be defined in this case.

        Q_s :
            Synchrotron tune. It can be defined only if longitudinal_mode is
            'linear'. If Q_s is provided, V_RF cannot be provided.

        alpha_mom_compaction :
            Momentum compaction factor (dL/L)/(dp)

        h_RF :
            Harmonic number. For multiple-harmonic RF systems this can be
            an array.

        V_RF : V
            RF voltage. For multiple-harmonic RF systems this can be
            an array.

        dphi_RF : rad
            Phase of the RF system with respect to the reference
            particle (z=0). For a single harmonic, in the absence of acceleration
            or energy losses:
             - above transition z = 0 is the stable fixed-point if dphi_RF = 0;
             - below transition z = 0 is the stable fixed-point if dphi_RF = pi.

        p_increment : kg m / s
            Acceleration, reference particle momentum change
            per turn.

        RF_at : 'middle', 'end_of_transverse'
            Position of the longitudinal map in the ring.

        wrap_z : True, False
            Wrap longitudinal position using the accelerator length.

        other_detuners :
            List of other detuners to be applied
            (default is other_detuners = []).
        """

        if use_cython:
            self.warns(
                "Cython modules no longer in use. Using Python module instead.\n"
            )

        self.charge = charge
        self.mass = mass
        self.p0 = p0
        self.optics_mode = optics_mode
        self.longitudinal_mode = longitudinal_mode

        self.one_turn_map = []

        # construct transverse map
        self._construct_transverse_map(
            optics_mode=optics_mode,
            circumference=circumference,
            n_segments=n_segments,
            name=name,
            s=s,
            alpha_x=alpha_x,
            beta_x=beta_x,
            D_x=D_x,
            alpha_y=alpha_y,
            beta_y=beta_y,
            D_y=D_y,
            accQ_x=accQ_x,
            accQ_y=accQ_y,
            Qp_x=Qp_x,
            Qp_y=Qp_y,
            app_x=app_x,
            app_y=app_y,
            app_xy=app_xy,
            other_detuners=other_detuners,
            use_cython=use_cython,
        )

        # construct longitudinal map
        self._construct_longitudinal_map(
            longitudinal_mode=longitudinal_mode,
            Q_s=Q_s,
            alpha_mom_compaction=alpha_mom_compaction,
            h_RF=np.atleast_1d(h_RF),
            V_RF=np.atleast_1d(V_RF),
            dphi_RF=np.atleast_1d(dphi_RF),
            p_increment=p_increment,
            RF_at=RF_at,
        )

        # add longitudinal wrapper and buncher
        if wrap_z:
            self._add_wrapper_and_buncher()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma ** -2)
        self._betagamma = np.sqrt(self.gamma ** 2 - 1)
        self._p0 = self.betagamma * self.mass * c

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self.gamma = 1.0 / np.sqrt(1 - value ** 2)

    @property
    def betagamma(self):
        return self._betagamma

    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value ** 2 + 1)

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, value):
        self.gamma = 1 / (c * self.mass) * np.sqrt(value ** 2 + self.mass ** 2 * c ** 2)

    @property
    def Q_x(self):
        return np.atleast_1d(self.transverse_map.accQ_x)[-1]

    @property
    def Q_y(self):
        return np.atleast_1d(self.transverse_map.accQ_y)[-1]

    @property
    def Q_s(self):
        return np.atleast_1d(self.longitudinal_map.Q_s)[-1]

    def track(self, bunch, verbose=False):
        for m in self.one_turn_map:
            if verbose:
                self.prints("Tracking through:\n" + str(m))
            m.track(bunch)

    def install_after_each_transverse_segment(self, element_to_add):
        """Attention: Do not add any elements which update the dispersion!"""
        one_turn_map_new = []
        for element in self.one_turn_map:
            one_turn_map_new.append(element)
            if element in self.transverse_map:
                one_turn_map_new.append(element_to_add)
        self.one_turn_map = one_turn_map_new

    def generate_6D_Gaussian_bunch(
        self, n_macroparticles, intensity, epsn_x, epsn_y, sigma_z
    ):
        """Generate a 6D Gaussian distribution of particles which is
        transversely matched to the Synchrotron. Longitudinally, the
        distribution is matched only in terms of linear focusing.
        For a non-linear bucket, the Gaussian distribution is cut along
        the separatrix (with some margin). It will gradually filament
        into the bucket. This will change the specified bunch length.
        """
        if self.longitudinal_mode == "linear":
            check_inside_bucket = lambda z, dp: np.array(len(z) * [True])
            Q_s = self.longitudinal_map.Q_s
        elif self.longitudinal_mode == "non-linear":
            bucket = self.longitudinal_map.get_bucket(
                gamma=self.gamma, mass=self.mass, charge=self.charge
            )
            check_inside_bucket = bucket.make_is_accepted(margin=0.05)
            Q_s = bucket.Q_s
        else:
            raise NotImplementedError("Something wrong with self.longitudinal_mode")

        eta = self.longitudinal_map.alpha_array[0] - self.gamma ** -2
        beta_z = np.abs(eta) * self.circumference / 2.0 / np.pi / Q_s
        sigma_dp = sigma_z / beta_z
        epsx_geo = epsn_x / self.betagamma
        epsy_geo = epsn_y / self.betagamma

        injection_optics = self.transverse_map.get_injection_optics()

        bunch = generators.ParticleGenerator(
            macroparticlenumber=n_macroparticles,
            intensity=intensity,
            charge=self.charge,
            mass=self.mass,
            circumference=self.circumference,
            gamma=self.gamma,
            distribution_x=generators.gaussian2D(epsx_geo),
            alpha_x=injection_optics["alpha_x"],
            beta_x=injection_optics["beta_x"],
            D_x=injection_optics["D_x"],
            distribution_y=generators.gaussian2D(epsy_geo),
            alpha_y=injection_optics["alpha_y"],
            beta_y=injection_optics["beta_y"],
            D_y=injection_optics["D_y"],
            distribution_z=generators.cut_distribution(
                generators.gaussian2D_asymmetrical(sigma_u=sigma_z, sigma_up=sigma_dp),
                is_accepted=check_inside_bucket,
            ),
        ).generate()

        return bunch

    def generate_6D_Gaussian_bunch_matched(
        self, n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=None, epsn_z=None
    ):
        """Generate a 6D Gaussian distribution of particles which is
        transversely as well as longitudinally matched.
        The distribution is found iteratively to exactly yield the
        given bunch length while at the same time being stationary in
        the non-linear bucket. Thus, the bunch length should amount
        to the one specificed and should not change significantly
        during the synchrotron motion.

        Requires self.longitudinal_mode == 'non-linear'
        for the bucket.
        """
        if self.longitudinal_mode == 'linear':
            assert(sigma_z is not None)
            bunch = self.generate_6D_Gaussian_bunch(n_macroparticles, intensity,
                    epsn_x, epsn_y, sigma_z)
        elif self.longitudinal_mode == "non-linear":
            epsx_geo = epsn_x / self.betagamma
            epsy_geo = epsn_y / self.betagamma

            injection_optics = self.transverse_map.get_injection_optics()

            bunch = generators.ParticleGenerator(
                macroparticlenumber=n_macroparticles,
                intensity=intensity,
                charge=self.charge,
                mass=self.mass,
                circumference=self.circumference,
                gamma=self.gamma,
                distribution_x=generators.gaussian2D(epsx_geo),
                alpha_x=injection_optics["alpha_x"],
                beta_x=injection_optics["beta_x"],
                D_x=injection_optics["D_x"],
                distribution_y=generators.gaussian2D(epsy_geo),
                alpha_y=injection_optics["alpha_y"],
                beta_y=injection_optics["beta_y"],
                D_y=injection_optics["D_y"],
                distribution_z=generators.RF_bucket_distribution(
                    self.longitudinal_map.get_bucket(gamma=self.gamma),
                    sigma_z=sigma_z,
                    epsn_z=epsn_z,
                ),
            ).generate()
        else:
            raise ValueError('Unknown longitudinal mode!')

        return bunch

    def _construct_transverse_map(
        self,
        optics_mode=None,
        circumference=None,
        n_segments=None,
        s=None,
        name=None,
        alpha_x=None,
        beta_x=None,
        D_x=None,
        alpha_y=None,
        beta_y=None,
        D_y=None,
        accQ_x=None,
        accQ_y=None,
        Qp_x=None,
        Qp_y=None,
        app_x=None,
        app_y=None,
        app_xy=None,
        other_detuners=None,
        use_cython=None,
    ):

        if optics_mode == "smooth":
            if circumference is None:
                raise ValueError(
                    "circumference has to be specified " 'if optics_mode = "smooth"'
                )
            if n_segments is None:
                raise ValueError(
                    "n_segments has to be specified " 'if optics_mode = "smooth"'
                )
            if s is not None:
                raise ValueError(
                    "s vector should not be provided " 'if optics_mode = "smooth"'
                )

            s = np.arange(0, n_segments + 1) * circumference / n_segments
            alpha_x = 0.0 * s
            beta_x = 0.0 * s + beta_x
            D_x = 0.0 * s + D_x
            alpha_y = 0.0 * s
            beta_y = 0.0 * s + beta_y
            D_y = 0.0 * s + D_y

        elif optics_mode == "non-smooth":
            if circumference is not None:
                raise ValueError(
                    "circumference should not be provided "
                    'if optics_mode = "non-smooth"'
                )
            if n_segments is not None:
                raise ValueError(
                    "n_segments should not be provided " 'if optics_mode = "non-smooth"'
                )
            if s is None:
                raise ValueError("s has to be specified " 'if optics_mode = "smooth"')

        else:
            raise ValueError("optics_mode not recognized")

        detuners = []
        if any(np.atleast_1d(Qp_x) != 0) or any(np.atleast_1d(Qp_y) != 0):
            detuners.append(Chromaticity(Qp_x, Qp_y))
        if app_x != 0 or app_y != 0 or app_xy != 0:
            detuners.append(AmplitudeDetuning(app_x, app_y, app_xy))
        detuners += other_detuners

        self.transverse_map = TransverseMap(
            s=s,
            alpha_x=alpha_x,
            beta_x=beta_x,
            D_x=D_x,
            alpha_y=alpha_y,
            beta_y=beta_y,
            D_y=D_y,
            accQ_x=accQ_x,
            accQ_y=accQ_y,
            detuners=detuners,
        )

        self.circumference = s[-1]
        self.transverse_map.n_segments = len(s) - 1

        if name is None:
            self.transverse_map.name = ["P_%d" % ip for ip in range(len(s) - 1)]
            self.transverse_map.name.append("end_ring")
        else:
            self.transverse_map.name = name

        for i_seg, m in enumerate(self.transverse_map):
            m.i0 = i_seg
            m.i1 = i_seg + 1
            m.s0 = self.transverse_map.s[i_seg]
            m.s1 = self.transverse_map.s[i_seg + 1]
            m.name0 = self.transverse_map.name[i_seg]
            m.name1 = self.transverse_map.name[i_seg + 1]
            m.beta_x0 = self.transverse_map.beta_x[i_seg]
            m.beta_x1 = self.transverse_map.beta_x[i_seg + 1]
            m.beta_y0 = self.transverse_map.beta_y[i_seg]
            m.beta_y1 = self.transverse_map.beta_y[i_seg + 1]

        # insert transverse map in the ring
        for m in self.transverse_map:
            self.one_turn_map.append(m)

    def _construct_longitudinal_map(
        self,
        longitudinal_mode=None,
        h_RF=None,
        V_RF=None,
        dphi_RF=None,
        alpha_mom_compaction=None,
        Q_s=None,
        p_increment=None,
        RF_at=None,
    ):

        if longitudinal_mode is None:
            return

        # Provide an RF bucket if possible
        if (h_RF[0] is not None
            and V_RF[0] is not None
            and dphi_RF[0] is not None
            and alpha_mom_compaction is not None
            and p_increment is not None
        ):

            self.rfbucket = RFBucket(
                charge=self.charge,
                mass=self.mass,
                circumference=self.circumference,
                gamma=self.gamma,
                alpha_array=np.atleast_1d(alpha_mom_compaction),
                p_increment=p_increment,
                voltage_list=V_RF,
                harmonic_list=h_RF,
                phi_offset_list=dphi_RF,
            )

        if RF_at == "middle":
            # compute the index of the element before which to insert
            # the longitudinal map
            if longitudinal_mode is not None:
                for insert_before, si in enumerate(self.transverse_map.s):
                    if si > 0.5 * self.circumference:
                        break
            insert_before -= 1
        elif RF_at == "end_of_transverse":
            insert_before = -1
        else:
            raise ValueError("RF_at=%s not recognized!")

        if longitudinal_mode == "linear":
            if V_RF[0] is not None and Q_s is not None:
                raise ValueError("Q_s and V_RF cannot be provided at the same time")

            if Q_s is None:
                try:
                    Q_s = self.rfbucket.Q_s
                except AttributeError:
                    raise AttributeError("Q_s not available!")

            self.longitudinal_map = LinearMap(
                np.atleast_1d(alpha_mom_compaction),
                self.circumference,
                Q_s,
                D_x=self.transverse_map.D_x[insert_before],
                D_y=self.transverse_map.D_y[insert_before],
            )

        elif longitudinal_mode == "non-linear":

            if Q_s is not None:
                raise ValueError(
                    "Q_s cannot be provided if longitudinal mode is non-linear"
                )

            self.longitudinal_map = RFSystems(
                charge=self.charge,
                mass=self.mass,
                circumference=self.circumference,
                gamma_reference=self.gamma,
                voltage_list=np.atleast_1d(V_RF),
                harmonic_list=np.atleast_1d(h_RF),
                phi_offset_list=np.atleast_1d(dphi_RF),
                alpha_array=np.atleast_1d(alpha_mom_compaction),
                p_increment=p_increment,
                D_x=self.transverse_map.D_x[insert_before],
                D_y=self.transverse_map.D_y[insert_before],
            )
        else:
            raise NotImplementedError("Something wrong with longitudinal_mode")

        if insert_before == -1:
            self.one_turn_map.append(self.longitudinal_map)
        else:
            self.one_turn_map.insert(insert_before, self.longitudinal_map)

    def _add_wrapper_and_buncher(self):
        """Add longitudinal z wrapping around the circumference as
        well as a UniformBinSlicer for bunching the beam.
        """
        if self.longitudinal_mode is None:
            return

        elif self.longitudinal_mode == "linear":
            raise ValueError("Not implemented!!!!")

        elif self.longitudinal_mode == "non-linear":
            bucket = self.longitudinal_map.get_bucket(
                gamma=self.gamma, mass=self.mass, charge=self.charge
            )
            harmonic = bucket.h[0]
            bucket_length = self.circumference / harmonic
            z_beam_center = (
                bucket.z_ufp_separatrix + bucket_length - self.circumference / 2.0
            )
            self.z_wrapper = LongWrapper(
                circumference=self.circumference, z0=z_beam_center
            )
            self.one_turn_map.append(self.z_wrapper)
            self.buncher = UniformBinSlicer(
                harmonic, z_cuts=(self.z_wrapper.z_min, self.z_wrapper.z_max)
            )

        else:
            raise NotImplementedError("Something wrong with longitudinal_mode")


""" The below doesn't work well... this we need to think of how to do it
properly. It does not seem to be a common problem in any case.


@deprecated('--> "BasicSynchrotron" will be removed '
            'in the near future. Use "Synchrotron" instead.\n')
class BasicSynchrotron(Synchrotron):
    pass
"""

# @deprecated_class('--> "BasicSynchrotron" will be removed '
#             'in the near future. Use "Synchrotron" instead.\n')
# class BasicSynchrotron(Synchrotron):
#     pass

# class BasicSynchrotron(Synchrotron):
#     @deprecated('"--> BasicSynchrotron" will be deprecated ' +
#                 'in the near future. Use "Synchrotron" instead.\n')
#     def __init__(self, *args, **kwargs):
#         Synchrotron.__init__(self, *args, **kwargs)


# @deprecated('--> "BasicSynchrotron" will be removed '
#             'in the near future. Use "Synchrotron" instead.\n')
# def BasicSynchrotron(*args, **kwargs):
#     return Synchrotron(*args, **kwargs)


import warnings


class BasicSynchrotron(Synchrotron):
    def __init__(self, *args, **kwargs):

        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            '\n\n*** DEPRECATED: "BasicSynchrotron" will be replaced in a future '
            'PyHEADTAIL release! You may want to use "Synchrotron" instead.',
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)

        Synchrotron.__init__(self, *args, **kwargs)
