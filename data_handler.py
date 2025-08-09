import pandas as pd
import numpy as np
from utils import binary_matrix


# For mock data
def generate_data(random_seed=42):
    np.random.seed(random_seed)

    def make_random_num_between(a, b, size=1, decimals=2):
        return np.round(np.random.uniform(a, b, size=size), decimals=decimals)

    bonds = ["B1", "B2", "B3", "B4", "B5", "B6"]

    market_data = {
        "Bond": bonds,
        "market price": make_random_num_between(1.0, 100.0, size=len(bonds)),
        "min trade": 100 * make_random_num_between(1.00, 10.00, size=len(bonds)),
        "max trade": 10e3 * make_random_num_between(1, 1e3, size=len(bonds)),
        "basket invent": 10e3 * make_random_num_between(1, 1e3, size=len(bonds)),
        "min increment": 100 * make_random_num_between(1.0, 10.0, size=len(bonds)),
    }
    market = pd.DataFrame(market_data)
    market.set_index("Bond", inplace=True)

    # Define characteristics (set J)
    characteristics = [
        "duration",
        "yield",
        "rating",
        "sector_financials",
        "liquidity_score",
    ]

    # Define risk buckets (set L)
    risk_buckets = {
        "Low risk": ["B1", "B2", "B3", "B4"],
        "High risk": ["B2", "B3", "B5", "B6"],
        "Balanced": ["B1", "B3", "B4", "B5"],
    }

    # Generate C_j: characteristics of each bond
    C_j_data = {
        "Bond": bonds,
        "duration": make_random_num_between(2.0, 7.0, len(bonds)),
        "yield": make_random_num_between(3.0, 7.0, len(bonds)),
        "rating": make_random_num_between(0, 1, len(bonds)),
        "sector_financials": np.random.choice([0, 1], size=len(bonds)),
        "liquidity_score": make_random_num_between(60, 100, len(bonds), decimals=1),
    }
    C_j = pd.DataFrame(C_j_data)
    C_j.set_index("Bond", inplace=True)

    # Target characteristics
    targets = {
        "Low risk": {
            "duration": 3.59,
            "yield": 4.17,
            "rating": 0.48,
            "sector_financials": 0.00,
            "liquidity_score": 77.40,
        },
        "High risk": {
            "duration": 4.27,
            "yield": 3.04,
            "rating": 0.70,
            "sector_financials": 1.00,
            "liquidity_score": 72.90,
        },
        "Balanced": {
            "duration": 3.99,
            "yield": 3.98,
            "rating": 0.74,
            "sector_financials": 1.00,
            "liquidity_score": 91.20,
        },
    }

    # Define target K_lj and bounds [b_lj, b'_lj] for each risk bucket and characteristic
    targets = []
    for bucket in risk_buckets:
        for char in characteristics:
            if char in ["duration", "yield"]:
                K = make_random_num_between(3.0, 6.0)[0]
            if char == "rating":
                K = make_random_num_between(0.3, 0.8)[0]
            if char == "sector_financials":
                K = np.random.randint(0, 2)
            if char == "liquidity_score":
                K = np.round(np.random.uniform(70, 95), 1)

            lower = (
                K - make_random_num_between(0.0, 0.2)[0] if isinstance(K, float) else K
            )
            upper = (
                K + make_random_num_between(0.0, 0.2)[0] if isinstance(K, float) else K
            )
            targets.append(
                {
                    "Risk Bucket": bucket,
                    "Characteristic": char,
                    "K_lj": K,
                    "Lower Bound (b_lj)": lower,
                    "Upper Bound (b_lj')": upper,
                }
            )
    K_lj = pd.DataFrame(targets)

    # Beta_cj: Contribution of each bond to the target of characteristic

    rows = []
    for bucket, bond_list in risk_buckets.items():
        for bond in bond_list:
            for _, row in K_lj[K_lj["Risk Bucket"] == "Balanced"].iterrows():
                charac = row["Characteristic"]
                target = row["K_lj"]
                value = C_j.loc[bond, charac]
                # Avoid division by zero
                B_cj = value / target if target != 0 else float(0)
                rows.append(
                    {
                        "Risk Bucket": bucket,
                        "Bond": bond,
                        "Characteristic": charac,
                        "B_cj": B_cj,
                    }
                )

    B_cj_df = pd.DataFrame(rows)

    m_vec = np.array(market["market price"])
    # mintrade_vec = np.array(market["min trade"])
    maxtrade_vec = np.array(market["max trade"])
    i_vec = np.array(market["basket invent"])
    delta_vec = np.array(market["min increment"])
    alpha_vec = (m_vec + np.min(np.array([maxtrade_vec, i_vec]), axis=0)) / (
        2 * delta_vec
    )

    y_vec = make_random_num_between(0, 1, size=len(bonds), decimals=8)
    x_vec = alpha_vec * y_vec

    K_lj_np = K_lj.to_numpy()
    K_lj_np = 800 * K_lj_np
    K_1j = K_lj_np[:, 2][:5]
    K_2j = K_lj_np[:, 2][5:10]
    K_3j = K_lj_np[:, 2][10:]
    B_cj_np = B_cj_df.to_numpy()
    B_cj_np_1 = B_cj_np[:, 3][:5]
    B_cj_np_2 = B_cj_np[:, 3][5:10]
    B_cj_np_3 = B_cj_np[:, 3][10:15]
    B_cj_np_4 = B_cj_np[:, 3][15:20]
    B_cj_np_5 = B_cj_np[:, 3][30:35]
    B_cj_np_6 = B_cj_np[:, 3][35:40]
    B_cj_rearrange = np.array(
        [B_cj_np_1, B_cj_np_2, B_cj_np_3, B_cj_np_4, B_cj_np_5, B_cj_np_6]
    )
    # B_c1=B_cj_rearrange[:,0]
    # B_c2=B_cj_rearrange[:,1]
    # B_c3=B_cj_rearrange[:,2]
    # B_c4=B_cj_rearrange[:,3]
    # B_c5=B_cj_rearrange[:,4]

    # beta_jc_np = np.random.uniform(low=0.7, high=1.3, size=(5, 6))
    rho_j = np.random.uniform(low=0.2, high=1.3, size=(5))
    return alpha_vec, B_cj_rearrange, rho_j, K_1j, K_2j, K_3j


def ob(z, alpha_vec, B_cj_rearrange, rho_j, K_1j, K_2j, K_3j):
    # z is a list [z1,z2,...]
    y_vec = (1 - z) / 2
    x_vec = alpha_vec * y_vec
    x_vec_1 = x_vec.copy()
    x_vec_1[4] = 0
    x_vec_1[5] = 0
    x_vec_2 = x_vec.copy()
    x_vec_2[1] = 0
    x_vec_2[3] = 0
    x_vec_3 = x_vec.copy()
    x_vec_3[1] = 0
    x_vec_3[5] = 0
    l1 = 0
    for i in range(5):
        l = rho_j[i] * (B_cj_rearrange[:, i] @ x_vec_1 - K_1j[i]) ** 2
        l1 = l1 + l
    l2 = 0
    for i in range(5):
        l = rho_j[i] * (B_cj_rearrange[:, i] @ x_vec_2 - K_2j[i]) ** 2
        l2 = l2 + l
    l3 = 0
    for i in range(5):
        l = rho_j[i] * (B_cj_rearrange[:, i] @ x_vec_3 - K_3j[i]) ** 2
        l3 = l3 + l
    f = l1 + l2 + l3
    return f / 1e9


def get_baseline(b=1):
    alpha_vec, B_cj_rearrange, rho_j, K_1j, K_2j, K_3j = generate_data()
    C_func = lambda z: ob(z, alpha_vec, B_cj_rearrange, rho_j, K_1j, K_2j, K_3j)
    objective = []
    for z1 in [-1, 1]:
        for z2 in [-1, 1]:
            for z3 in [-1, 1]:
                for z4 in [-1, 1]:
                    for z5 in [-1, 1]:
                        for z6 in [-1, 1]:
                            z = np.array([z1, z2, z3, z4, z5, z6])
                            objective.append(C_func(z))
    objective = np.array(objective)
    prob = np.exp(-b * objective)
    Z = np.sum(prob)
    prob = prob / Z
    return objective.min(), prob


## For real data (Vanguard 31bonds)
def vg_ob_qubo(x_bin_repr: np.array, datasets: np.array):

    return x_bin_repr.T @ datasets @ x_bin_repr


def vg_constr_cost(
    x_bin_repr: np.array,
    qubo_ob_mat: np.array,
    ob_c: np.float64,
    penalty: np.float64,
    constr_A: np.array,
    constr_b: np.array,
):
    return (
        vg_ob_qubo(x_bin_repr, qubo_ob_mat)
        + ob_c
        + penalty * np.sum(np.maximum(constr_b - constr_A @ x_bin_repr, 0) ** 2)
    )


def vg_ob_qubo_baseline(objective: np.array):
    n_qubits = objective.shape[0]
    binary_mat = binary_matrix(n_qubits)

    bolz_exp = []
    for bin in binary_mat:
        bolz_exp.append(vg_ob_qubo(bin, objective))

    prob = np.exp(-np.array(bolz_exp))
    norm_factor = np.sum(prob)

    return bolz_exp.min(), prob / norm_factor


def vg_constr_cost_baseline(
    qubo_ob_mat: np.array,
    ob_c: np.float64,
    penalty: np.float64,
    constr_A: np.array,
    constr_b: np.array,
):
    n_qubits = qubo_ob_mat.shape[0]
    binary_mat = binary_matrix(n_qubits)

    constr_cost = []
    for bin in binary_mat:
        constr_cost.append(
            vg_constr_cost(
                x_bin_repr=bin,
                qubo_ob_mat=qubo_ob_mat,
                ob_c=ob_c,
                penalty=penalty,
                constr_A=constr_A,
                constr_b=constr_b,
            )
        )

    norm_cost = np.array(constr_cost) / np.array(constr_cost).max()
    prob = np.exp(-np.array(norm_cost))
    norm_factor = np.sum(prob)

    baseline = np.min(constr_cost)
    prob_baseline = prob / norm_factor

    return baseline, prob_baseline
