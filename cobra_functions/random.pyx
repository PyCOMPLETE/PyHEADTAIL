from cython_gsl cimport *


def gsl_random(bunch):

    cdef double[::1] x = bunch.x
    cdef double[::1] xp = bunch.xp
    cdef double[::1] y = bunch.y
    cdef double[::1] yp = bunch.yp
    cdef double[::1] dz = bunch.dz
    cdef double[::1] dp = bunch.dp

    cdef int i, n_particles = len(x)

    cdef gsl_rng_type* T
    cdef gsl_rng* r

    gsl_rng_env_setup()
    T = gsl_rng_default
    r = gsl_rng_alloc(T)

    for i in xrange(n_particles):
        x[i], xp[i], y[i], yp[i], dz[i], dp[i] = get_rng_gaussian(r)[:7]

    return x, xp, y, yp, dz, dp

cdef get_rng_gaussian(gsl_rng* r):
 
    cdef list s = [0] * 6
    cdef double tmp[6]

    for i in xrange(6):
        s[i] = gsl_ran_gaussian(r, 1.0)

    return s

def gsl_quasirandom(bunch):

    cdef double[::1] x = bunch.x
    cdef double[::1] xp = bunch.xp
    cdef double[::1] y = bunch.y
    cdef double[::1] yp = bunch.yp
    cdef double[::1] dz = bunch.dz
    cdef double[::1] dp = bunch.dp

    cdef int i, n_particles = len(x)

    cdef gsl_qrng* r
    r = gsl_qrng_alloc(gsl_qrng_niederreiter_2, 6)
    gsl_qrng_init(r)
    
    for i in xrange(n_particles):
        x[i], xp[i], y[i], yp[i], dz[i], dp[i] = get_qrng_gaussian(r)[:7]
        # x[i], xp[i], y[i], yp[i], dz[i], dp[i] = get_qrng_uniform(r)[:7]

    return x, xp, y, yp, dz, dp

cdef get_qrng_uniform(gsl_qrng* r):

    cdef list s = [0] * 6
    cdef double tmp[6]

    gsl_qrng_get(r, tmp)
    for i in xrange(6):
        s[i] = tmp[i] * 2 - 1

    return s

cdef get_qrng_gaussian(gsl_qrng* r):
 
    cdef list s = [0] * 6
    cdef double tmp[6]

# //             // Classical Box-Muller
# //             for (int i=0; i<n_elements; i++)
# //             {
# //                 gsl_qrng_get(qr, tmp);
# //                 phi = 2*M_PI*tmp[0];
# //                 radius = log(1-tmp[1]);
# // //                 phasespace[i][ix] = epsn*sqrt(-2*radius)*sin(phi);
# // //                 phasespace[i][ix+3] = sigma*sqrt(-2*radius)*cos(phi);
# //                 u[i] = epsn*sqrt(-2*radius)*sin(phi);
# //                 v[i] = sigma*sqrt(-2*radius)*cos(phi);
# //             }

    # quasi-random sequence is correlated -> rejection method can not work!
    # Polar Box-Muller
    cdef int i
    cdef double p, q, radius
    for i in xrange(3):
        gsl_qrng_get(r, tmp)

        p = tmp[i * 2] * 2 - 1
        q = tmp[i * 2 + 1] * 2 - 1
        radius = p ** 2 + q ** 2
        while (radius == 0 or radius >= 1.0):
            gsl_qrng_get(r, tmp)

            p = tmp[i * 2] * 2 - 1
            q = tmp[i * 2 + 1] * 2 - 1
            radius = p ** 2 + q ** 2

        s[i * 2] = p * sqrt(-2.0 * log(radius) / radius)
        s[i * 2 + 1] = q * sqrt(-2.0 * log(radius) / radius)

    return s




'''
class RandomBase
{
public:
    RandomBase(std::string distribution): distribution(distribution) { };
    virtual ~RandomBase() { };

    std::vector<double> get_random_numbers()
    {
        if (distribution=="uniform")
            return get_uniform();
        else if (distribution=="gaussian")
            return get_gaussian();
        else
            return std::vector<double>(6, 0);
    }

    std::string distribution;

private:
    virtual std::vector<double> get_uniform() = 0;
    virtual std::vector<double> get_gaussian() = 0;
};


class Random_gsl_rng : public RandomBase
{
public:
    Random_gsl_rng(std::string distribution): RandomBase(distribution)
    {
        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
    }
    virtual ~Random_gsl_rng() { gsl_rng_free(r); }

private:
    const gsl_rng_type* T;
    gsl_rng* r;

    std::vector<double> get_uniform()
    {
        size_t n_dims = 6;
        std::vector<double> s(n_dims);
        double tmp[n_dims];

        for (size_t i=0; i<n_dims; i++)
            s[i] = gsl_rng_uniform(r) - 0.5;

        return s;
    }

    std::vector<double> get_gaussian()
    {
        size_t n_dims = 6;
        std::vector<double> s(n_dims);
        double tmp[n_dims];

        for (size_t i=0; i<n_dims; i++)
            s[i] = gsl_ran_gaussian(r, 1.0);

        return s;
    }
};


class Random_gsl_qrng : public RandomBase
{
public:
    Random_gsl_qrng(std::string distribution): RandomBase(distribution)
    {
        T = gsl_qrng_halton;
        r = gsl_qrng_alloc(T, 6);
        gsl_qrng_init(r);
    }
    ~Random_gsl_qrng() { gsl_qrng_free(r); }

private:
    const gsl_qrng_type* T;
    gsl_qrng* r;

    std::vector<double> get_uniform()
    {
        size_t n_dims = 6;
        std::vector<double> s(n_dims);
        double tmp[n_dims];

        gsl_qrng_get(r, tmp);
        for (size_t i=0; i<n_dims; i++)
            s[i] = tmp[i] - 0.5;

        return s;
    }

    std::vector<double> get_gaussian()
    {
        size_t n_dims = 6;
        std::vector<double> s(n_dims);
        double tmp[n_dims];

        // Polar Box-Muller
        double p, q, radius;
        for (size_t i=0; i<n_dims/2; i++)
        {
            do
            {
                gsl_qrng_get(r, tmp);

                p = 2.0 * tmp[i * 2] - 1.0;
                q = 2.0 * tmp[i * 2 + 1] - 1.0;
                radius = p*p + q*q;
            }
            while (radius >= 1.0);

            s[i * 2] = p * sqrt(-2.0 * log(radius) / radius);
            s[i * 2 + 1] = q * sqrt(-2.0 * log(radius) / radius);
        }

        return s;
    }
};
'''
