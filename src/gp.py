import operator, random, math
from deap import gp, base, creator, tools
from .policy import GPPolicy, build_baseline
from .sim import run_episode
from .io_drive import load_instance
from .features import FEATURE_ORDER


def protected_div(a, b):
    try:
        return a / b if abs(b) > 1e-12 else a
    except Exception:
        return a


def build_pset(n_args):
    pset = gp.PrimitiveSet("F", n_args)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addPrimitive(abs, 1)
    pset.addPrimitive(lambda x: math.tanh(x), 1, name="tanh")
    # rename args theo thực tự feature
    for i, name in enumerate(FEATURE_ORDER):
        pset.renameArguments(**{f"ARG{i}": name})
    return pset


def compile_ind(toolbox, ind):
    return toolbox.compile(expr=ind)


def build_toolbox(cfg):
    random.seed(cfg.get("seed", 0))
    n = len(FEATURE_ORDER)
    pset = build_pset(n)
    # avoid recreating the same classes (DEAP warns if recreated)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=cfg["gp"]["max_depth"])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset  # keep handle for later mutation calls
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=cfg["gp"]["max_depth"])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=50))
    return toolbox


def evaluate_ind(cfg, ind_r, ind_s, req_df, T_ref):
    toolbox = build_toolbox(cfg)
    Rf = compile_ind(toolbox, ind_r)
    Sf = compile_ind(toolbox, ind_s)
    pol = GPPolicy(Rf, Sf)
    stats, _ = run_episode(cfg, pol, req_df)
    lam = float(cfg["objective"]["lambda_w"])
    F = lam * (stats["makespan"] / max(1e-9, T_ref)) + (1.0 - stats["served"] / stats["total"])
    return F, stats


def train_gphh(cfg):
    # load data & baseline T_ref
    req_df, _ = load_instance(cfg)
    base_pol = build_baseline()
    base_stats, _ = run_episode(cfg, base_pol, req_df)
    T_ref = max(1e-9, base_stats["makespan"])

    tb = build_toolbox(cfg)
    pop_size = cfg["gp"]["pop_size"]
    inter = cfg["gp"]["intermed_size"]
    gens = cfg["gp"]["generations"]
    k = cfg["gp"]["tournament_k"]

    # each individual is a tuple (routing_tree, sequencing_tree)
    pop = [(tb.individual(), tb.individual()) for _ in range(pop_size)]

    def fitness(pair):
        indR, indS = pair
        F, _ = evaluate_ind(cfg, indR, indS, req_df, T_ref)
        return (F,)

    def tournament_select(pop, fits, count, tournsize):
        out = []
        for _ in range(count):
            cand_idx = random.sample(range(len(pop)), tournsize)
            best_i = min(cand_idx, key=lambda i: fits[i][0])
            out.append(pop[best_i])
        return out

    fits = list(map(fitness, pop))
    for gen in range(gens):
        order = sorted(range(len(pop)), key=lambda i: fits[i][0])
        elites = [pop[order[0]], pop[order[1]]] if len(pop) >= 2 else [pop[order[0]]]
        parents = elites + tournament_select(pop, fits, max(0, inter - len(elites)), k)

        offspring = []
        while len(offspring) < pop_size:
            a = random.choice(parents)
            b = random.choice(parents)
            ar, as_ = gp.PrimitiveTree(a[0]), gp.PrimitiveTree(a[1])
            br, bs = gp.PrimitiveTree(b[0]), gp.PrimitiveTree(b[1])
            if random.random() < cfg["gp"]["rc"]:
                gp.cxOnePoint(ar, br)
                gp.cxOnePoint(as_, bs)
            if random.random() < cfg["gp"]["rm"]:
                gp.mutUniform(ar, expr=tb.expr_mut, pset=tb.pset)
            if random.random() < cfg["gp"]["rm"]:
                gp.mutUniform(as_, expr=tb.expr_mut, pset=tb.pset)
            offspring.append((ar, as_))

        # elitism: keep best from previous generation
        best_idx = min(range(len(pop)), key=lambda i: fits[i][0])
        best = pop[best_idx]

        pop = offspring
        fits = list(map(fitness, pop))

        worst_idx = max(range(len(pop)), key=lambda i: fits[i][0])
        pop[worst_idx] = best
        fits[worst_idx] = fitness(best)

        best_now = min(fits, key=lambda f: f[0])[0]
        print(f"[Gen {gen+1}/{gens}] best F = {best_now:.4f}")

    best_idx = min(range(len(pop)), key=lambda i: fits[i][0])
    best_pair = pop[best_idx]
    _, stats = evaluate_ind(cfg, best_pair[0], best_pair[1], req_df, T_ref)
    print("Best stats:", stats)
    return best_pair


def build_gp_policy_from_pair(cfg, pair):
    tb = build_toolbox(cfg)
    Rf = tb.compile(pair[0])
    Sf = tb.compile(pair[1])
    return GPPolicy(Rf, Sf)
