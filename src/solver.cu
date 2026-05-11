#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWOPI = 2.0 * PI;

struct Vec4 {
    double x[4];
};

struct Basis {
    Vec4 v1;
    Vec4 v2;
};

struct StateSens {
    Vec4 z;      // z = (theta, phi, lambda1, lambda2), relative to one equilibrium
    Vec4 zp;     // derivative dz/dpsi, where psi parameterizes the local stable circle
    double J;    // accumulated running cost along the backward stable-manifold orbit
    bool ok;
};

struct Guess {
    double d2;
    double psi;
    double tau;
};

struct Candidate {
    Result r;
    double cost;
    double residual;
    bool ok;
};

static inline double sqr(double x) {
    return x * x;
}

static inline double norm4(const Vec4& v) {
    return std::sqrt(
        sqr(v.x[0]) + sqr(v.x[1]) + sqr(v.x[2]) + sqr(v.x[3])
    );
}

static inline bool finite4(const Vec4& v) {
    return std::isfinite(v.x[0]) &&
           std::isfinite(v.x[1]) &&
           std::isfinite(v.x[2]) &&
           std::isfinite(v.x[3]);
}

static inline Vec4 add_scaled(const Vec4& a, const Vec4& b, double h) {
    Vec4 r{};
    for (int i = 0; i < 4; ++i) {
        r.x[i] = a.x[i] + h * b.x[i];
    }
    return r;
}

/*
   Minimized Hamiltonian system.

   State equations:
       theta_dot = phi
       phi_dot   = sin(theta) - alpha*phi - lambda2*cos(theta)^2

   Costate equations:
       lambda1_dot = -sin(theta) - lambda2*cos(theta)
                     - lambda2^2 sin(theta) cos(theta)

       lambda2_dot = -phi - lambda1 + alpha*lambda2

   This comes from u* = -lambda2*cos(theta).
*/
static Vec4 rhs_forward(const Vec4& z, double alpha) {
    const double th = z.x[0];
    const double ph = z.x[1];
    const double l1 = z.x[2];
    const double l2 = z.x[3];

    (void)l1;

    const double s = std::sin(th);
    const double c = std::cos(th);

    Vec4 f{};
    f.x[0] = ph;
    f.x[1] = s - alpha * ph - l2 * c * c;
    f.x[2] = -s - l2 * c - l2 * l2 * s * c;
    f.x[3] = -ph - l1 + alpha * l2;
    return f;
}

static double running_cost(const Vec4& z) {
    const double th = z.x[0];
    const double ph = z.x[1];
    const double l2 = z.x[3];

    const double u = -l2 * std::cos(th);
    return (1.0 - std::cos(th)) + 0.5 * ph * ph + 0.5 * u * u;
}

// Backward continuation on the stable manifold: z_tau = -f(z).
static Vec4 rhs_backward(const Vec4& z, double alpha) {
    Vec4 f = rhs_forward(z, alpha);
    for (int i = 0; i < 4; ++i) {
        f.x[i] = -f.x[i];
    }
    return f;
}

/*
   Variational equation for dz/dpsi.

   If z_tau = g(z) = -f(z), then
       (dz/dpsi)_tau = Dg(z) (dz/dpsi).

   This lets Newton use a real Jacobian of the map
       (psi,tau) -> (theta,phi),
   rather than finite differences or a lookup table.
*/
static Vec4 jac_backward_times(const Vec4& z, const Vec4& w, double alpha) {
    const double th = z.x[0];
    const double l2 = z.x[3];

    const double s = std::sin(th);
    const double c = std::cos(th);
    const double c2 = c * c;
    const double cos2 = c * c - s * s;

    // Derivatives of the forward vector field f.
    const double df10 = c + 2.0 * l2 * s * c;
    const double df11 = -alpha;
    const double df13 = -c2;

    const double df20 = -c + l2 * s - l2 * l2 * cos2;
    const double df23 = -c - 2.0 * l2 * s * c;

    Vec4 out{};

    // out = -Df(z) * w.
    out.x[0] = -w.x[1];
    out.x[1] = -(df10 * w.x[0] + df11 * w.x[1] + df13 * w.x[3]);
    out.x[2] = -(df20 * w.x[0] + df23 * w.x[3]);
    out.x[3] = -(-w.x[1] - w.x[2] + alpha * w.x[3]);

    return out;
}

static Vec4 rk4_state_step(const Vec4& z, double h, double alpha) {
    Vec4 k1 = rhs_backward(z, alpha);
    Vec4 k2 = rhs_backward(add_scaled(z, k1, 0.5 * h), alpha);
    Vec4 k3 = rhs_backward(add_scaled(z, k2, 0.5 * h), alpha);
    Vec4 k4 = rhs_backward(add_scaled(z, k3, h), alpha);

    Vec4 out{};
    for (int i = 0; i < 4; ++i) {
        out.x[i] = z.x[i] + h * (
            k1.x[i] + 2.0 * k2.x[i] + 2.0 * k3.x[i] + k4.x[i]
        ) / 6.0;
    }

    return out;
}

static StateSens rhs_ext(const StateSens& y, double alpha) {
    StateSens r{};
    r.z = rhs_backward(y.z, alpha);
    r.zp = jac_backward_times(y.z, y.zp, alpha);
    r.J = running_cost(y.z);
    r.ok = true;
    return r;
}

static StateSens add_scaled_ext(const StateSens& a, const StateSens& b, double h) {
    StateSens r = a;

    for (int i = 0; i < 4; ++i) {
        r.z.x[i]  = a.z.x[i]  + h * b.z.x[i];
        r.zp.x[i] = a.zp.x[i] + h * b.zp.x[i];
    }

    r.J = a.J + h * b.J;
    r.ok = true;
    return r;
}

static StateSens rk4_ext_step(const StateSens& y, double h, double alpha) {
    StateSens k1 = rhs_ext(y, alpha);
    StateSens k2 = rhs_ext(add_scaled_ext(y, k1, 0.5 * h), alpha);
    StateSens k3 = rhs_ext(add_scaled_ext(y, k2, 0.5 * h), alpha);
    StateSens k4 = rhs_ext(add_scaled_ext(y, k3, h), alpha);

    StateSens out = y;

    for (int i = 0; i < 4; ++i) {
        out.z.x[i] = y.z.x[i] + h * (
            k1.z.x[i] + 2.0 * k2.z.x[i] + 2.0 * k3.z.x[i] + k4.z.x[i]
        ) / 6.0;

        out.zp.x[i] = y.zp.x[i] + h * (
            k1.zp.x[i] + 2.0 * k2.zp.x[i] + 2.0 * k3.zp.x[i] + k4.zp.x[i]
        ) / 6.0;
    }

    out.J = y.J + h * (
        k1.J + 2.0 * k2.J + 2.0 * k3.J + k4.J
    ) / 6.0;

    out.ok = finite4(out.z) &&
             finite4(out.zp) &&
             std::isfinite(out.J) &&
             norm4(out.z) < 1.0e10 &&
             norm4(out.zp) < 1.0e12;

    return out;
}

/*
   Stable eigenspace at the equilibrium.

   The linearized Hamiltonian matrix has two stable eigenvalues. For this
   particular 4-by-4 structure, the stable eigenvalues can be written in
   closed form, so we do not need to depend on Eigen in the submitted code.
*/
static Vec4 eigenvector_for(double lambda, double alpha) {
    Vec4 v{};

    v.x[0] = 1.0;
    v.x[1] = lambda;
    v.x[3] = 1.0 - alpha * lambda - lambda * lambda;
    v.x[2] = -lambda + (alpha - lambda) * v.x[3];

    double n = norm4(v);
    if (n == 0.0) {
        n = 1.0;
    }

    for (int i = 0; i < 4; ++i) {
        v.x[i] /= n;
    }

    return v;
}

static Basis stable_basis(double alpha) {
    const double b = alpha * alpha + 3.0;
    const double disc = std::max(0.0, b * b - 8.0);

    const double mu_big   = 0.5 * (b + std::sqrt(disc));
    const double mu_small = 0.5 * (b - std::sqrt(disc));

    Basis B{};
    B.v1 = eigenvector_for(-std::sqrt(mu_big), alpha);
    B.v2 = eigenvector_for(-std::sqrt(mu_small), alpha);
    return B;
}

static Vec4 initial_state(const Basis& B, double psi) {
    const double eps = 1.0e-5;

    Vec4 z{};
    const double cp = std::cos(psi);
    const double sp = std::sin(psi);

    for (int i = 0; i < 4; ++i) {
        z.x[i] = eps * (cp * B.v1.x[i] + sp * B.v2.x[i]);
    }

    return z;
}

static StateSens initial_ext(const Basis& B, double psi) {
    const double eps = 1.0e-5;

    StateSens y{};
    const double cp = std::cos(psi);
    const double sp = std::sin(psi);

    for (int i = 0; i < 4; ++i) {
        y.z.x[i]  = eps * ( cp * B.v1.x[i] + sp * B.v2.x[i]);
        y.zp.x[i] = eps * (-sp * B.v1.x[i] + cp * B.v2.x[i]);
    }

    // Quadratic approximation to the value near the equilibrium.
    y.J = 0.5 * (y.z.x[0] * y.z.x[2] + y.z.x[1] * y.z.x[3]);
    y.ok = true;
    return y;
}

static StateSens manifold_point(
    const Basis& B,
    double alpha,
    double psi,
    double tau,
    double h
) {
    StateSens y = initial_ext(B, psi);

    if (tau < 0.0) {
        tau = 0.0;
    }

    int n = static_cast<int>(tau / h);

    for (int i = 0; i < n; ++i) {
        y = rk4_ext_step(y, h, alpha);
        if (!y.ok) {
            return y;
        }
    }

    const double rem = tau - n * h;
    if (rem > 1.0e-14) {
        y = rk4_ext_step(y, rem, alpha);
    }

    return y;
}

/*
   Coarse stable-manifold patch search.

   We parameterize the local stable eigenspace by psi and integrate backward
   for time tau. This creates a patch of the global stable manifold. The
   closest point in (theta,phi) gives the initial guess for Newton refinement.
*/
static Guess coarse_guess(const Basis& B, double alpha, double q, double phi) {
    const int npsi = 420;
    const int ntau = 620;

    const double R = std::sqrt(q * q + phi * phi);
    const double taumax = std::max(
        18.0,
        std::min(44.0, 13.0 + 1.45 * R + 0.35 * std::fabs(phi))
    );

    const double dtau = taumax / static_cast<double>(ntau);

    Guess best{
        std::numeric_limits<double>::infinity(),
        0.0,
        0.0
    };

    for (int i = 0; i < npsi; ++i) {
        const double psi = TWOPI * static_cast<double>(i) /
                           static_cast<double>(npsi);

        Vec4 z = initial_state(B, psi);

        for (int j = 0; j <= ntau; ++j) {
            const double d2 = sqr(z.x[0] - q) + sqr(z.x[1] - phi);

            if (d2 < best.d2) {
                best = {d2, psi, dtau * static_cast<double>(j)};
            }

            if (j < ntau) {
                z = rk4_state_step(z, dtau, alpha);

                if (!finite4(z) || norm4(z) > 1.0e10) {
                    break;
                }
            }
        }
    }

    return best;
}

/*
   Newton refinement for:
       F(psi,tau) = (theta(psi,tau)-q, phi(psi,tau)-phi_target) = 0.

   The first Jacobian column is dz/dpsi, obtained from the variational equation.
   The second Jacobian column is dz/dtau = backward_rhs(z).
*/
static bool newton_refine(
    const Basis& B,
    double alpha,
    double q,
    double phi,
    double& psi,
    double& tau,
    double h
) {
    for (int it = 0; it < 32; ++it) {
        StateSens y = manifold_point(B, alpha, psi, tau, h);

        if (!y.ok) {
            return false;
        }

        const double r0 = y.z.x[0] - q;
        const double r1 = y.z.x[1] - phi;
        const double nr = std::sqrt(r0 * r0 + r1 * r1);

        if (nr < 2.0e-11) {
            return true;
        }

        Vec4 gt = rhs_backward(y.z, alpha);

        const double a00 = y.zp.x[0];
        const double a01 = gt.x[0];
        const double a10 = y.zp.x[1];
        const double a11 = gt.x[1];

        const double det = a00 * a11 - a01 * a10;

        if (std::fabs(det) < 1.0e-14 || !std::isfinite(det)) {
            return false;
        }

        const double dpsi = ( a11 * r0 - a01 * r1) / det;
        const double dtau = (-a10 * r0 + a00 * r1) / det;

        bool improved = false;

        const double factors[] = {
            1.0, 0.7, 0.5, 0.3, 0.15,
            0.08, 0.04, 0.02, 0.01, 0.005
        };

        for (double fac : factors) {
            const double ps2 = psi - fac * dpsi;
            double ta2 = tau - fac * dtau;

            if (ta2 < 0.0) {
                ta2 = 0.0;
            }

            StateSens y2 = manifold_point(B, alpha, ps2, ta2, h);

            if (!y2.ok) {
                continue;
            }

            const double e0 = y2.z.x[0] - q;
            const double e1 = y2.z.x[1] - phi;
            const double ne = std::sqrt(e0 * e0 + e1 * e1);

            if (ne < nr) {
                psi = ps2;
                tau = ta2;
                improved = true;
                break;
            }
        }

        if (!improved) {
            return false;
        }
    }

    return false;
}

static Candidate solve_one_branch(
    const Basis& B,
    double alpha,
    double q,
    double phi
) {
    Guess g = coarse_guess(B, alpha, q, phi);

    const int npsi = 420;
    const double R = std::sqrt(q * q + phi * phi);

    const double taumax = std::max(
        18.0,
        std::min(44.0, 13.0 + 1.45 * R + 0.35 * std::fabs(phi))
    );

    const double dtau = taumax / 620.0;

    std::vector<std::pair<double, double>> seeds;
    seeds.emplace_back(g.psi, g.tau);

    for (double dp : {
        TWOPI / npsi,
        -TWOPI / npsi,
        2.0 * TWOPI / npsi,
        -2.0 * TWOPI / npsi
    }) {
        seeds.emplace_back(g.psi + dp, g.tau);
    }

    for (double dt : {
        dtau,
        -dtau,
        2.0 * dtau,
        -2.0 * dtau,
        4.0 * dtau,
        -4.0 * dtau
    }) {
        seeds.emplace_back(g.psi, std::max(0.0, g.tau + dt));
    }

    Candidate best{};
    best.ok = false;
    best.cost = std::numeric_limits<double>::infinity();
    best.residual = std::numeric_limits<double>::infinity();

    for (auto seed : seeds) {
        double psi = seed.first;
        double tau = seed.second;

        bool ok = newton_refine(B, alpha, q, phi, psi, tau, 0.010);

        if (ok) {
            ok = newton_refine(B, alpha, q, phi, psi, tau, 0.0030);
        }

        if (ok) {
            ok = newton_refine(B, alpha, q, phi, psi, tau, 0.0012);
        }

        if (!ok) {
            continue;
        }

        StateSens y = manifold_point(B, alpha, psi, tau, 0.00065);

        if (!y.ok) {
            continue;
        }

        const double res = std::sqrt(
            sqr(y.z.x[0] - q) + sqr(y.z.x[1] - phi)
        );

        if (res > 2.0e-7 || y.J < -1.0e-9 || !std::isfinite(y.J)) {
            continue;
        }

        if (y.J < best.cost) {
            best.ok = true;
            best.cost = y.J;
            best.residual = res;
            best.r = {
                y.z.x[2],
                y.z.x[3],
                std::max(0.0, y.J)
            };
        }
    }

    return best;
}

static Result linear_near_origin(double theta, double phi, double alpha) {
    /*
       Last-resort fallback only. In normal use, solve() returns a stable-
       manifold/Newton solution. This prevents pathological failure from
       producing NaNs.
    */

    const double k = std::round(theta / TWOPI);
    const double q = theta - TWOPI * k;

    Basis B = stable_basis(alpha);

    const double a = B.v1.x[0];
    const double b = B.v2.x[0];
    const double c = B.v1.x[1];
    const double d = B.v2.x[1];

    const double det = a * d - b * c;

    if (std::fabs(det) < 1.0e-14) {
        return {
            q,
            phi,
            0.5 * (q * q + phi * phi)
        };
    }

    const double y1 = ( d * q - b * phi) / det;
    const double y2 = (-c * q + a * phi) / det;

    const double l1 = B.v1.x[2] * y1 + B.v2.x[2] * y2;
    const double l2 = B.v1.x[3] * y1 + B.v2.x[3] * y2;

    return {
        l1,
        l2,
        std::max(0.0, 0.5 * (q * l1 + phi * l2))
    };
}

} // namespace

Result solve(double theta, double phi, double alpha) {
    if (std::fabs(theta) < 1.0e-14 && std::fabs(phi) < 1.0e-14) {
        return {0.0, 0.0, 0.0};
    }

    Basis B = stable_basis(alpha);

    /*
       The pendulum has equivalent equilibria theta = 2*pi*k. For states with
       large angular velocity, the lowest-cost trajectory may settle in the
       next well rather than the nearest one. Therefore we try several terminal
       equilibria and select the converged branch with minimal cost.
    */
    const int k0 = static_cast<int>(std::llround(theta / TWOPI));

    Candidate best{};
    best.ok = false;
    best.cost = std::numeric_limits<double>::infinity();
    best.residual = std::numeric_limits<double>::infinity();

    for (int dk = -4; dk <= 4; ++dk) {
        const int k = k0 + dk;
        const double q = theta - TWOPI * static_cast<double>(k);

        const double qlimit = 12.0 + 0.55 * std::fabs(phi);
        if (std::fabs(q) > qlimit) {
            continue;
        }

        Candidate c = solve_one_branch(B, alpha, q, phi);

        if (c.ok && c.cost < best.cost) {
            best = c;
        }
    }

    if (best.ok) {
        return best.r;
    }

    return linear_near_origin(theta, phi, alpha);
}
