# --- built in ---
# --- 3rd party ---
import torch
# --- my modlue ---
from rlchemy.lib import utils as rl_utils

__all__ = [
  'regression',
  'orthogonal'
]

def regression(
  err,
  loss_type: str = 'l2',
  w = None,
  huber: bool = None,
  delta: float = 1.0
):
  """Compute regression loss (L1, L2 or Huber) from error

  L1:
    L = |err|
  L2:
    L = |err|^2
  Huber:
    L ={ 1/2 * err^2, if |err| <= delta
       { delta * (|err| - delta/2), otherwise

  Args:
    err (torch.Tensor): error.
    w (torch.Tensor, optional): loss weights. Defaults to None.
    loss_type (str, optional): loss type, l1, l2 or huber.
      Defaults to 'l2'.
    huber (bool, optional): whether to use huber loss. Defaults to None.
    delta (float, optional): huber rate. Defaults to 1.0.
  
  Returns:
    torch.Tensor: regression loss.
  """
  # use huber loss if huber is True
  if huber is not None:
    loss_type = 'huber' if huber else loss_type
  # compute loss
  loss_type = loss_type.lower()
  if loss_type == 'l1':
    losses = torch.abs(err)
  elif loss_type == 'l2':
    losses = err ** 2.
  elif loss_type == 'huber':
    losses = torch.where(torch.abs(err) <= delta,
      0.5 * (err ** 2.),
      delta * torch.abs(err) - 0.5 * (delta ** 2.)
    )
  else:
    raise NotImplementedError(f"Loss type `{loss_type}` not implemented.")
  # weighting losses
  if w is not None:
    w = w.to(dtype=torch.float32)
    losses = losses * w
  return losses

def orthogonal(model, loss_type='l2'):
  """Orthogonal regularization
  from the equation (3) of arxiv:1809.11096

  L1 = ||W^T W * (1-eye)||
  L2 = ||W^T W * (1-eye)||^2

  Args:
    model (nn.Module): module to regularize.
    loss_type (str, optional): regularization type, l1 or l2.
      Defaults to 'l2'.
  
  Returns:
    torch.Tensor: regularization loss
  """
  loss_type = loss_type.lower()
  # compute orthogonal regularization loss
  def _orth_reg(param):
    param_flat = param.view(param.shape[0], -1)
    ww = torch.mm(param_flat, torch.t(param_flat))
    eye = torch.eye(param_flat.shape[0], device=ww.device)
    reg = ww * (1-eye)
    return torch.sum(regression(reg, loss_type=loss_type))
  # sum over all params' orth reg losses
  with torch.enable_grad():
    loss = sum([
      _orth_reg(param)
      for name, param in model.named_parameters()
      if 'bias' not in name
    ])
  return loss