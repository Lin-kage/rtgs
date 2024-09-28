
import torch
import plotly.graph_objects as go


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

eval_factor = 1. / pow(2. * torch.pi, 3./2.)


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def get_cov_3D(scales, rotations):
    L = build_scaling_rotation(scales, rotations)
    
    # print(f"\nL:{L.shape}\n {L}")
    
    return L @ L.transpose(-2, -1)


def eval_gaussian_3d(means, scales, rotations, opacities, shs, x):
    """
        Args:
            x: Tensor [M*3], M is the number of input points
    """
    
    # print(f"\nscales: {scales.shape}\n {scales}")
    # print(f"\rotations: {rotations.shape}\n {rotations}")
    
    cov3D = get_cov_3D(scales, rotations)   # [N, 3, 3]
    
    # print(f"\ncov3D: {cov3D.shape}\n {cov3D}")
    
    vecs = x[..., None, :] - means[None, ...]      # [M, N, 3]
    vecs_norm = vecs / vecs.norm(dim=-1, keepdim=True)
    
    # print(f"\nvec: {vecs.shape}\n {vecs}")
    
    matmul = (vecs[..., None, :] @ cov3D @ vecs[..., None, :].transpose(-2,-1)).reshape(x.shape[0], means.shape[0])  # [M, N]
    
    # print(f"\nmatmul: {matmul.shape}\n {matmul}")
    
    factor = eval_factor / torch.det(cov3D) * opacities  # [N]
    
    # print(f"\nfactor: {factor.shape}\n {factor}")
    
    # print(f"\nshs: {shs.shape}\n{shs}")
    
    sh = torch.clamp_min(eval_sh(4, shs, vecs_norm), 0)
    
    # print(f"\nsh: {sh.shape}\n{sh}")
    
    value = factor * torch.exp(-.5 * matmul) # * sh  # [M, N]
    
    # print(f"\nvalue: {value.shape}\n {value}") 
    
    value = value.sum(-1)
    return value  # [M]
    
    
def plot_3d(volume, res, points):
    X, Y, Z = torch.meshgrid(torch.linspace(0, 5, res), torch.linspace(0, 5, res), torch.linspace(0, 5, res), indexing='xy')
    
    X = X.detach().numpy()
    Y = Y.detach().numpy()
    Z = Z.detach().numpy()
    volume = volume.cpu().detach().numpy()
    
    # Z = 5 - Z
    # X = 5 - X
    
    points = points.cpu().detach().numpy()
    
    fig = go.Figure(data=[go.Volume(
		x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
		value=volume.flatten(),
		opacity=0.05,
		surface_count=10,
		),
        go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers")                
                          ])
    fig.update_layout(scene_xaxis_showticklabels=False,
					scene_yaxis_showticklabels=False,
					scene_zaxis_showticklabels=False)

    fig.show()
    