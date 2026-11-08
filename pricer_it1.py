import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def bs_call(S0, K, r, sigma, T):
    d1 = ((np.log(S0/K)) + (r + (sigma**2/2))*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0*norm.cdf(d1)-norm.cdf(d2)*K*np.exp(-r*T)
    #norm cdf finds N(d1) and N(D2)

def simulate_terminal_prices(S0, r, sigma, T, n_paths, rng, antithetic= False):
    if antithetic:
        assert n_paths % 2 == 0, "antithetic should be even"
        #antithetic needs and even number of paths, so if its not it alerts us
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
        #ties all Z values in array to a corresponding negative one
    else:
        Z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

def mc_call(S0, K, r, sigma, T, n_paths, rng, antithetic = False):
    S_T = simulate_terminal_prices(S0, r, sigma, T, n_paths, rng, antithetic)
    payoffs = np.maximum(S_T - K, 0)
    if antithetic:
        half = n_paths // 2
        samples = (payoffs[:half] + payoffs[half:]) / 2
        #averages out each antithetic pairs payoff
    else:
        samples = payoffs
    price = np.exp(-r*T) * samples.mean()
    se = np.exp(-r*T) * samples.std() / np.sqrt(len(samples))

    return price, se
    
def convergence_study(S0, K, r, sigma, T, sizes, n_reps=20 , seed=0):
    rng = np.random.default_rng(seed)
    #creates a random number generator seeded at 0
    bs = bs_call(S0, K, r, sigma, T)
    results = []
    for anti in (False, True):
    # repeats once with antithetic false then true
        error = []
        for n in sizes:
            errs = []
            for _ in range(n_reps):
                errs.append(mc_call(S0, K, r, sigma, T, n, rng, anti)[0] - bs)
                #send the monte carlo results to the errs array
            error.append(np.sqrt(np.mean(np.square(errs))))
            # RMS error for each value in errs
        results.append(error)
    return results[0], results[1]
        
 
 
def plot_convergence(sizes, err_plain,err_anti, path="convergence.png"):
    sizes = np.asarray(sizes)
    err_plain = np.asarray(err_plain)
    err_anti = np.asarray(err_anti)

    c = np.mean(err_plain * np.sqrt(sizes))
    # fit c so that the reference line is c/sqrt(n)
    ref = c/np.sqrt(sizes)

    plt.loglog(sizes, err_plain, "o-", label="MC error")
    plt.loglog(sizes, ref, "--", label="1/sqrt(n)")
    plt.loglog(sizes, err_anti, "s-", label="Antithetic")
    plt.xlabel("Number of paths")
    plt.ylabel("Rms error against Black-Scholes")
    plt.title("Monte Carlo convergence for a Europan call option")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.show()


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    rng = np.random.default_rng(42)
    n_paths = 1_000_000

    bs = bs_call(S0, K, r, sigma, T)
    mc, se = mc_call(S0, K, r, sigma, T, n_paths, rng)
    diff = mc-bs
    sizes = [10**2,10**3,10**4,10**5,10**6]
    print("The Black Scholes equation estimates the price to be $",round(bs,4))
    print("Our Monte Carlo simulation estimates the price at $", round(mc,4))
    print("They have a difference of ", diff)

    print("Within 3 standard errors:", bool(np.abs(diff) < 3 * se))
    #checks whether difference is within 3 standard errors

    mc_a, se_a = mc_call(S0, K, r, sigma, T, n_paths, rng, antithetic=True)
    #repeats monte carlo simulation but with antithetics
    print("SE plain:", se, "SE antithetic:", se_a, "reduction:", 1 - se_a/se)
    

    err_plain, err_anti = convergence_study(S0, K, r, sigma, T, sizes)
    plot_convergence(sizes, err_plain, err_anti)