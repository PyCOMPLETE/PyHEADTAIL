class TransverseDamper(object):

    def __init__(self, dampingrate_x, dampingrate_y, phase_x_deg=90, phase_y_deg=90,
                 beta_x=1, beta_y=1):
        self.gain_x = 2./dampingrate_x
        self.gain_y = 2./dampingrate_y

        self.phase_x = phase_x_deg * np.pi/180
        self.phase_y = phase_y_deg * np.pi/180

        self.beta_x = beta_x
        self.beta_y = beta_y

    def track(self, beam):
        beam.xp -= self.gain_x * (np.cos(self.phase_x)*beam.mean_x()/self.beta_x +
                                  np.sin(self.phase_x)*beam.mean_xp())
        beam.yp -= self.gain_y * (np.cos(self.phase_y)*beam.mean_y()/self.beta_y +
                                  np.sin(self.phase_y)*beam.mean_yp())
        # beam.xp -= self.gain_x * beam.mean_xp()
	# beam.yp -= self.gain_y * beam.mean_yp()
