import torch
import plotly.graph_objects as go


def plot_3d(volume: torch.Tensor, res, points: torch.Tensor=torch.zeros([1,3]), reverse=False):
    X, Y, Z = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing='xy')
    
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()
    
    volume = volume.cpu().detach().numpy()
    
    if reverse:
      K = X
      X = Y
      Y = K
    
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
    