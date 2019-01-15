import pandas as pd
import numpy as np
from scipy import stats
import math
from scipy.ndimage import rotate
from scipy.spatial.distance import pdist, squareform
import numba


@numba.jit(nopython=True)
def norm_mat(v_mat, axis=1):
    return(np.sqrt(v_mat[:, 0]**2 + v_mat[:, 1]**2 + v_mat[:, 2]**2))


@numba.jit(nopython=True)
def norm_vec2(v, axis=1):
    return(np.sqrt(v[0]**2 + v[1]**2))


@numba.jit(nopython=True)
def norm_vec3(v, axis=1):
    return(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2))


@numba.jit(nopython=True)
def multivariate_normal_pdf(x, mu, L, detL):
    diff2 = (x - mu)**2
    dLd = np.dot(diff2, L @ diff2)
    Z = detL / np.power(2*math.pi, diff2.shape[0]/2)
    prob = Z*math.exp(-dLd)
    return(prob)


class DataManager:
    """
    This class manage the data.
    In particular, provide r and v for one time point.
    """
    def file_load(self, fname):
        self.df = pd.read_csv(fname)

    def set_time(self, t):
        selected_df = self.df.loc[self.df.t == t]
        pre_column_dict = {
            "vx": "pvx",
            "vy": "pvy",
            "vz": "pvz"
        }
        pre_df = self.df.loc[self.df.t == (t-1)].rename(
            pre_column_dict,
            axis=1)[["pvx", "pvy", "pvz", "track"]]
        merged_df = selected_df.merge(pre_df, on='track').drop_duplicates(
            ["x", "y", "z"])
        self.r = merged_df[["x", "y", "z"]].values
        self.v = merged_df[["vx", "vy", "vz"]].values
        self.pre_v = merged_df[["pvx", "pvy", "pvz"]].values
        self.dist_mat = squareform(pdist(self.r))

    def get_v(self):
        return(self.v)

    def get_r(self):
        return(self.r)

    def get_pre_v(self):
        return(self.pre_v)


class Decider:
    """
    This class calculate difference between simulation and real data,
    and decide whether simulation results is acceptable.
    """
    def __init__(self, length, v, max_eps=0.1, rate=0.9):
        """
        Initialize epx vec, which represents accetance threshold for each round
        Aquire the real data velocity used in evaluation
        Waring:: Eps vec should be considered later
        """
        self.length = length
        self.eps_vec = max_eps * np.power(rate, np.arange(length))
        self.v = v

    def decide(self, v, t):
        """
        Decide accetance
        """
        eps = self.eps_vec[t]
        acceptance = np.mean(norm_mat(self.v - v, axis=1)) < eps
        return(acceptance)

    def get_length(self):
        return(self.length)


class SampleRegister:
    """
    This class manage sampled parameters and weight for each of them.
    """
    def __init__(self, prior, transition):
        """
        Set prior function and transition function of parameters.
        Initialize vector of new_theta and new_w
        """
        self.prior = prior
        self.transition = transition
        self.new_w_vec = np.array([])
        self.new_theta_vec = np.array([])
        self.new_num = 0

    def register_new(self, new_theta, t):
        """
        Register a new parameter.
        Simultaneously it calculate weight for the new parameter,
        and register it.
        """
        self.new_theta_vec = np.append(self.new_theta_vec, new_theta)
        if t > 0:
            w = self.prior(new_theta)/np.sum(
                [w * self.transition(theta, new_theta)
                 for theta, w in zip(self.theta_vec, self.w_vec)])
        else:
            w = 1
        self.new_w_vec = np.append(self.new_w_vec, w)
        self.new_num = self.new_w_vec.shape[0]

    def load_new(self):
        """
        Refresh theta_vec and w_vec
        """
        self.theta_vec = self.new_theta_vec
        self.new_theta_vec = np.array([])
        self.w_vec = self.new_w_vec
        self.new_w_vec = np.array([])
        self.new_num = 0

    def choice(self):
        """
        Randomly choose parameters based on weight
        """
        return(
            np.random.choice(
                self.theta_vec,
                size=1, p=self.w_vec/np.sum(self.w_vec))[0])

    def get_new_num(self):
        return(self.new_num)


class PriorSampler:
    """
    Sample parameters from prior distribution
    """
    def __init__(self, theta_key_vec, boundary_dict, K_dict, L_base):
        self.theta_key_vec = theta_key_vec
        self.boundary_dict = boundary_dict
        self.K_dict = K_dict
        self.L_base = L_base
        point_num = K_dict[theta_key_vec[0]].shape[0]
        self.point_num_vec = np.full(point_num, 1.0)

    def sample_param(self):
        """
        Sample parameters based on prior distribution
        Mean of each parameter sampled from uniform distribution
        Local values are sampled from multivariate gaussian,
        which reflect spatial location of each cell
        """
        theta = {}
        for theta_key in self.theta_key_vec:
            theta[theta_key] = {}
            theta[theta_key]["mean"] = np.random.uniform(
                self.boundary_dict[theta_key][0],
                self.boundary_dict[theta_key][1],
                size=1)[0]
            # loca parameters sampled from mean
            theta[theta_key]["local"] = np.abs(np.random.multivariate_normal(
                theta[theta_key]["mean"] * self.point_num_vec,
                ((0.1*theta[theta_key]["mean"])**2)*self.K_dict[theta_key]))
        return(theta)

    def calc_prior_prob(self, theta):
        """
        Calculate the likelihood of theta based on prior distribution
        """
        prob = 1
        for theta_key in self.theta_key_vec:
            interval = self.boundary_dict[theta_key][1] - self.boundary_dict[theta_key][0]
            mean_prob = 1.0/interval
            L = self.L_base/((0.1*theta[theta_key]["mean"])**2)
            local_prob = multivariate_normal_pdf(
                theta[theta_key]["local"],
                theta[theta_key]["mean"] * self.point_num_vec,
                L, np.linalg.det(L))
            prob *= mean_prob * local_prob
        return(prob)


class SequentialSampler(PriorSampler):
    """
    Parameter sampling based on previous parameter
    """
    def __init__(self, theta_key_vec, var_dict, K_dict, L_base):
        self.theta_key_vec = theta_key_vec
        self.var_dict = var_dict
        self.K_dict = K_dict
        self.L_base = L_base
        point_num = K_dict[theta_key_vec[0]].shape[0]
        self.point_num_vec = np.full(point_num, 1.0)

    def sample_param(self, pre_theta):
        """
        Sample parameters from previous theta
        """
        theta = {}
        for theta_key in self.theta_key_vec:
            theta[theta_key] = {}
            # mean parameter sampled from previous mean
            theta[theta_key]["mean"] = np.abs(np.random.normal(
                pre_theta[theta_key]["mean"],
                pre_theta[theta_key]["mean"]*0.1,
                size=1)[0])
            # local parameters sampled from previous local
            diff_mean = theta[theta_key]["mean"] - pre_theta[theta_key]["mean"]
            theta[theta_key]["local"] = np.abs(np.random.multivariate_normal(
                pre_theta[theta_key]["local"] + diff_mean,
                ((0.1*pre_theta[theta_key]["mean"])**2)*self.K_dict[theta_key]))
        return(theta)

    def calc_transition_prob(self, pre_theta, theta):
        """
        Calculate transition probability from pre_theta to theta
        """
        prob = 1
        for theta_key in self.theta_key_vec:
            # mean parameter sampled from previous mean
            mean_prob = stats.norm.pdf(
                theta[theta_key]["mean"],
                loc=pre_theta[theta_key]["mean"],
                scale=pre_theta[theta_key]["mean"]*0.1)
            # local parameters sampled from previous local
            diff_mean = theta[theta_key]["mean"] - pre_theta[theta_key]["mean"]
            L = self.L_base/((0.1*pre_theta[theta_key]["mean"])**2)
            local_prob = multivariate_normal_pdf(
                theta[theta_key]["local"],
                pre_theta[theta_key]["local"] + diff_mean,
                L, np.linalg.det(L))
            prob *= mean_prob * local_prob
        return(prob)


def chart2polar3d(x):
    """
    Convert chart coordinate to polar coordinate
    """
    r = norm_vec3(x)
    if r == 0:
        theta = 0
        phi = 0
    else:
        theta = math.acos(x[2]/r)
        rxy = norm_vec2(x[:2])
        phi = np.sign(x[1]) * math.acos(x[0]/rxy)
    polar = np.array([r, theta, phi])
    return(polar)


def rotate3d_x(theta):
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])
    return(rotate_x)


def rotate3d_y(theta):
    rotate_y = np.array([[np.cos(theta), 0, -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0, np.cos(theta)]])
    return(rotate_y)


def rotate3d_z(theta):
    rotate_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    return(rotate_z)


def get_random_dev_vec(x, dev_angle):
    """
    Get random vector whose angle deviation to x is dev_angle
    """
    z_vec = np.array([0, 0, 1])
    random_theta = np.random.uniform(-math.pi, math.pi)
    dev_z_vec = rotate3d_z(random_theta) @ rotate3d_y(-dev_angle) @ z_vec
    polar = chart2polar3d(x)
    random_dev_vec = rotate3d_z(polar[2]) @ rotate3d_y(-polar[1]) @ dev_z_vec
    return(random_dev_vec)


def get_maximum_neghbor(x, r_max):
    neighbor_list = [
        [j for j in range(x.shape[0])
         if norm_vec3(x[i, :] - x[j, :]) < r_max and i != j]
        for i in range(x.shape[0])]
    return(neighbor_list)


class Simulator:
    """
    This class conduct simulation
    """
    def __init__(self, x, pre_v, max_r=30):
        self.x = x
        self.pre_v = pre_v
        self.cell_num = x.shape[0]
        self.maximum_neghbor = get_maximum_neghbor(x, max_r)

    def conduct(self, theta):
        """
        Conduct simulation for all cell
        This return (cell_num, 3) matrix which represents
        estimated velocity of all cells
        """
        est_v = np.stack([self.conduct_each_cell(cell, theta)
                          for cell in range(self.cell_num)])
        return(est_v)

    def conduct_each_cell(self, cell, theta):
        """
        Conduct simulation for each cell
        This return (3) array which represents
        estimated velocity of one cell
        """
        self_v = self.get_self_v(cell, theta)
        neighbor_vec = self.get_neighbor(cell, theta)
        outer_f = np.zeros(3)
        for neighbor in neighbor_vec:
            outer_f += self.get_action(cell, neighbor, theta)
        est_each_v = self_v + outer_f
        return(est_each_v)

    def get_self_v(self, cell, theta):
        """
        Calculate self propelled speed
        """
        v0 = theta["v0"]["local"][cell]
        eta = theta["eta"]["local"][cell]
        xi = np.random.uniform(-eta*math.pi, eta*math.pi)
        base_direction = self.pre_v[cell]/norm_vec3(self.pre_v[cell])
        direction = get_random_dev_vec(base_direction, xi)
        self_v = v0 * direction
        return(self_v)

    def get_action(self, cell, neighbor, theta):
        """
        Calculate action from neighbor
        """
        re_cell = theta["re"]["local"][cell]
        re_neighbor = theta["re"]["local"][neighbor]
        dr0_cell = theta["dr0"]["local"][cell]
        dr0_neighbor = theta["dr0"]["local"][neighbor]
        r_cell = self.x[cell]
        r_neighbor = self.x[neighbor]
        dist_neighbor_cell = norm_vec3(r_cell - r_neighbor)
        if dist_neighbor_cell > 0.1:
            direction_neighbor_cell = (r_cell - r_neighbor)/dist_neighbor_cell
        else:
            direction_neighbor_cell = np.zeros(r_cell.shape)
        fa = theta["fa"]["local"][cell]
        fr = theta["fr"]["local"][cell]
        Re = re_cell + re_neighbor
        dR0 = dr0_cell + dr0_neighbor
        if dist_neighbor_cell < Re:
            outer_f = direction_neighbor_cell*fr*(Re - dist_neighbor_cell)/Re
        else:
            outer_f = direction_neighbor_cell*fa*(Re - dist_neighbor_cell)/(dR0)
        if dist_neighbor_cell < 0.1:
            outer_f = np.zeros(r_cell.shape)
        return(outer_f)

    def get_neighbor(self, cell, theta):
        """
        Get neighbor of a cell paid attention to
        """
        x_cell = self.x[cell]
        re_cell = theta["re"]["local"][cell]
        dr0_cell = theta["dr0"]["local"][cell]
        r0_cell = re_cell + dr0_cell
        neighbor_vec = np.array([])
        for i in self.maximum_neghbor[cell]:
            x_i = self.x[i]
            re_i = theta["re"]["local"][i]
            dr0_i = theta["dr0"]["local"][i]
            r0_i = re_i + dr0_i
            dist_cell_i = norm_vec3(x_cell - x_i)
            if dist_cell_i < r0_cell + r0_i and i != cell:
                neighbor_vec = np.append(neighbor_vec, i)
        return(neighbor_vec.astype(int))


class LcpMain:
    def main(self, dm, dc, sr, ps, ss, sim, sample_num, max_iter=1.0e5):
        theta = {}
        count = 0
        for t in range(dc.get_length()):
            print("t: " + str(t))
            sr.load_new()
            pre_theta = theta
            while sr.get_new_num() < sample_num and count <= max_iter:
                if t == 0:
                    theta = ps.sample_param()
                else:
                    theta = ss.sample_param(pre_theta)
                v = sim.conduct(theta)
                if dc.decide(v, t):
                    sr.register_new(theta, t)
                    print("num: " + str(sr.get_new_num()))
                count += 1
            if count > max_iter:
                break
            pre_theta = theta
        return(sr, v)
